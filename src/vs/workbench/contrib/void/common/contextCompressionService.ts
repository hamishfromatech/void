/*--------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { LLMChatMessage } from './sendLLMMessageTypes.js';
import { ITokenCountingService } from './tokenCountingService.js';

// Tools whose calls read file/system content (ported from a-coder-cli's
// compaction file-op tracking). Their paths are surfaced in compression
// summaries so the post-compression agent still knows what was inspected.
const FILE_READ_TOOLS = new Set([
	'read_file', 'outline_file', 'ls_dir', 'get_dir_tree',
	'search_pathnames_only', 'search_for_files', 'search_in_file',
	'read_lint_errors', 'fast_context', 'codebase_search',
]);
// Tools whose calls mutate the workspace — their paths are critical context.
const FILE_MODIFY_TOOLS = new Set([
	'create_file_or_folder', 'delete_file_or_folder', 'edit_file', 'edit_files', 'rewrite_file',
]);
// Param keys that may carry a filesystem path on tool calls.
const PATH_PARAM_KEYS = ['uri', 'path', 'filepath', 'folder_uri', 'file_path'] as const;

/**
 * Configuration for context compression with rolling window support
 */
export interface CompressionConfig {
	/** Target percentage of context window to use (0-1) - lower means more aggressive compression */
	targetUsage: number;
	/** Minimum number of recent messages to always keep (preserves recency) */
	keepLastNMessages: number;
	/** Whether to summarize old messages vs removing them */
	enableSummarization: boolean;
	/** Maximum length for tool results before truncation */
	maxToolResultLength: number;
	/** Reserved tokens for system message and output */
	reservedTokens: number;
	/** Emergency fallback: if still over limit, keep only last N messages */
	emergencyKeepLastN: number;
}

/**
 * Default compression configuration - optimized for rolling window approach
 */
export const DEFAULT_COMPRESSION_CONFIG: CompressionConfig = {
	targetUsage: 0.85, // Use 85% of context window (more aggressive)
	keepLastNMessages: 10, // Keep last 10 messages (5 turns) for better context
	enableSummarization: true,
	maxToolResultLength: 50000, // Allow larger tool results for file operations (was 1500, too aggressive)
	reservedTokens: 8192, // Reserve 8K tokens for system + output (adjustable per model)
	emergencyKeepLastN: 4, // In emergency, keep only last 4 messages
};

/**
 * Statistics about the compression operation
 */
export interface CompressionStats {
	originalTokens: number;
	originalMessageCount: number;
	targetTokens: number;
	finalTokens: number;
	finalMessageCount: number;
	messagesRemoved: number;
	messagesSummarized: number;
	toolResultsTruncated: number;
	compressionRatio: number; // Percentage of original tokens retained
}

/**
 * Service for compressing message context using a rolling window approach
 *
 * Strategy (in order):
 * 1. Truncate large tool results
 * 2. Summarize middle messages (preserving recent and system)
 * 3. Remove oldest messages if still over limit (rolling window)
 * 4. Emergency: keep only the most recent messages
 */
export class ContextCompressionService {
	constructor(
		private tokenCountingService: ITokenCountingService
	) { }

	/**
	 * Extract { id?, name } for every tool call in a message, across provider
	 * formats (OpenAI tool_calls, Anthropic tool_use blocks, Gemini functionCall
	 * parts). Returns an empty array for non-assistant / non-tool-call messages.
	 */
	private extractToolCallsFromMessage(msg: LLMChatMessage): Array<{ id?: string; name: string }> {
		const calls: Array<{ id?: string; name: string }> = [];
		if ('role' in msg && msg.role === 'assistant') {
			const m = msg as LLMChatMessage & { tool_calls?: Array<{ id?: string; function?: { name?: string } }>; content?: unknown };
			if (Array.isArray(m.tool_calls)) {
				for (const tc of m.tool_calls) {
					const name = tc.function?.name;
					if (name) calls.push({ id: tc.id, name });
				}
			}
			if (Array.isArray(m.content)) {
				for (const part of m.content as Array<{ type?: string; id?: string; name?: string }>) {
					if (part?.type === 'tool_use' && part.name) calls.push({ id: part.id, name: part.name });
				}
			}
		}
		if ('parts' in msg && Array.isArray((msg as LLMChatMessage & { parts?: Array<{ functionCall?: { name?: string } }> }).parts)) {
			for (const part of (msg as LLMChatMessage & { parts?: Array<{ functionCall?: { name?: string } }> }).parts!) {
				if (part?.functionCall?.name) calls.push({ name: part.functionCall.name });
			}
		}
		return calls;
	}

	/** True when the message carries tool results (any provider format). */
	private isToolResultMessage(msg: LLMChatMessage): boolean {
		if ('role' in msg) {
			if (msg.role === 'tool') return true;
			if (msg.role === 'user' && Array.isArray((msg as LLMChatMessage & { content?: unknown }).content)) {
				return ((msg as LLMChatMessage & { content: Array<{ type?: string }> }).content)
					.some(part => part?.type === 'tool_result');
			}
		}
		if ('parts' in msg && Array.isArray((msg as LLMChatMessage & { parts?: Array<{ functionResponse?: unknown }> }).parts)) {
			return ((msg as LLMChatMessage & { parts: Array<{ functionResponse?: unknown }> }).parts)
				.some(part => !!part?.functionResponse);
		}
		return false;
	}

	/** True when the assistant message carries tool calls (any provider format). */
	private isAssistantWithToolCalls(msg: LLMChatMessage): boolean {
		if ('role' in msg && msg.role === 'assistant') {
			const m = msg as LLMChatMessage & { tool_calls?: unknown[]; content?: unknown };
			if (Array.isArray(m.tool_calls) && m.tool_calls.length > 0) return true;
			if (Array.isArray(m.content)) {
				return (m.content as Array<{ type?: string }>).some(part => part?.type === 'tool_use');
			}
		}
		if ('parts' in msg && Array.isArray((msg as LLMChatMessage & { parts?: Array<{ functionCall?: unknown }> }).parts)) {
			return ((msg as LLMChatMessage & { parts: Array<{ functionCall?: unknown }> }).parts)
				.some(part => !!part?.functionCall);
		}
		return false;
	}

	/**
	 * Find the smallest valid cut index >= desiredIdx (ported from a-coder-cli's
	 * findValidCutPoints): never orphan a tool result from its call, and never
	 * drop an assistant tool-call sequence while keeping its results. If no
	 * valid boundary exists, returns messages.length (nothing gets summarized).
	 */
	private findValidCutPoint(messages: LLMChatMessage[], desiredIdx: number): number {
		for (let i = Math.max(1, desiredIdx); i < messages.length; i++) {
			const msg = messages[i];
			const prev = messages[i - 1];
			// Invalid: cutting here orphans this tool result from its call.
			if (this.isToolResultMessage(msg)) continue;
			// Invalid: the previous (removed) message carries tool calls whose
			// results live in the kept region.
			if (this.isAssistantWithToolCalls(prev)) continue;
			return i;
		}
		return messages.length;
	}

	/**
	 * Remove tool calls and tool results that no longer have their counterpart
	 * (ported from a-coder-cli compaction): providers reject tool results without
	 * a preceding call AND assistant tool-calls without a following result. Runs
	 * after any operation that drops messages from the middle of a sequence.
	 */
	private dropOrphanedToolMessages(messages: LLMChatMessage[]): LLMChatMessage[] {
		// Pass 1: collect the ids/names that still exist on the other side.
		const callIds = new Set<string>();
		const resultIds = new Set<string>();
		const callNames = new Set<string>();
		const resultNames = new Set<string>();
		for (const msg of messages) {
			for (const call of this.extractToolCallsFromMessage(msg)) {
				if (call.id) callIds.add(call.id);
				callNames.add(call.name);
			}
			if ('role' in msg && msg.role === 'tool') {
				const id = (msg as LLMChatMessage & { tool_call_id?: string }).tool_call_id;
				if (id) resultIds.add(id);
			} else if ('role' in msg && msg.role === 'user' && Array.isArray((msg as LLMChatMessage & { content?: unknown }).content)) {
				for (const part of (msg as LLMChatMessage & { content: Array<{ type?: string; tool_use_id?: string }> }).content) {
					if (part?.type === 'tool_result' && part.tool_use_id) resultIds.add(part.tool_use_id);
				}
			} else if ('parts' in msg && Array.isArray((msg as LLMChatMessage & { parts?: Array<{ functionResponse?: { name?: string } }> }).parts)) {
				for (const part of (msg as LLMChatMessage & { parts: Array<{ functionResponse?: { name?: string } }> }).parts) {
					if (part?.functionResponse?.name) resultNames.add(part.functionResponse.name);
				}
			}
		}

		// Pass 2: rebuild, keeping only paired calls/results.
		const out: LLMChatMessage[] = [];
		for (const msg of messages) {
			if ('role' in msg && msg.role === 'assistant') {
				const m = msg as LLMChatMessage & { tool_calls?: Array<{ id?: string; function?: { name?: string } }>; content?: unknown };
				let hadCalls = false;
				let keptCalls: Array<{ id?: string; function?: { name?: string } }> | undefined;
				if (Array.isArray(m.tool_calls)) {
					hadCalls = m.tool_calls.length > 0;
					keptCalls = m.tool_calls.filter(tc => !!tc.id && resultIds.has(tc.id));
				}
				let hadToolUse = false;
				let keptContent: Array<{ type?: string; id?: string }> | undefined;
				if (Array.isArray(m.content)) {
					const contentArr = m.content as Array<{ type?: string; id?: string }>;
					hadToolUse = contentArr.some(part => part?.type === 'tool_use');
					keptContent = contentArr.filter(part => part?.type !== 'tool_use' || (!!part.id && resultIds.has(part.id)));
				}
				if (hadCalls && keptCalls!.length === 0 && (!keptContent || keptContent.length === 0)) continue; // drop pure-call message
				if (hadToolUse && keptContent && keptContent.length === 0 && keptCalls!.length === 0) continue; // drop pure-call message
				out.push({
					...msg,
					...(keptCalls !== undefined ? { tool_calls: keptCalls } : {}),
					...(keptContent !== undefined ? { content: keptContent } : {}),
				} as LLMChatMessage);
				continue;
			}
			if ('role' in msg && msg.role === 'tool') {
				const id = (msg as LLMChatMessage & { tool_call_id?: string }).tool_call_id;
				if (id && !callIds.has(id)) continue; // orphaned result
				out.push(msg);
				continue;
			}
			if ('role' in msg && msg.role === 'user' && Array.isArray((msg as LLMChatMessage & { content?: unknown }).content)) {
				const contentArr = (msg as LLMChatMessage & { content: Array<{ type?: string; tool_use_id?: string }> }).content;
				const hasResults = contentArr.some(part => part?.type === 'tool_result');
				if (hasResults) {
					const keptContent = contentArr.filter(part => part?.type !== 'tool_result' || (!!part.tool_use_id && callIds.has(part.tool_use_id)));
					if (keptContent.length === 0) continue; // orphaned results only
					out.push({ ...msg, content: keptContent } as LLMChatMessage);
					continue;
				}
				out.push(msg);
				continue;
			}
			if ('parts' in msg && Array.isArray((msg as LLMChatMessage & { parts?: Array<{ functionCall?: { name?: string } | unknown; functionResponse?: { name?: string } | unknown }> }).parts)) {
				const parts = (msg as LLMChatMessage & { parts: Array<{ functionCall?: { name?: string }; functionResponse?: { name?: string } }> }).parts;
				const isCall = (p: { functionCall?: unknown }) => !!p?.functionCall;
				const isResponse = (p: { functionResponse?: unknown }) => !!p?.functionResponse;
				if (parts.some(isCall) || parts.some(isResponse)) {
					const keptParts = parts.filter(p => {
						if (isCall(p)) return resultNames.has(p.functionCall!.name!);
						if (isResponse(p)) return callNames.has(p.functionResponse!.name!);
						return true;
					});
					if (keptParts.length === 0) continue;
					out.push({ ...msg, parts: keptParts } as LLMChatMessage);
					continue;
				}
				out.push(msg);
				continue;
			}
			out.push(msg);
		}
		return out;
	}

	/**
	 * Extract workspace file operations from tool calls in messages (ported from
	 * a-coder-cli's compaction file-op tracking). Used to preserve file context
	 * through compression summaries.
	 */
	private extractFileOperations(messages: LLMChatMessage[]): { filesRead: string[]; filesModified: string[] } {
		const filesRead = new Set<string>();
		const filesModified = new Set<string>();

		for (const msg of messages) {
			for (const call of this.extractToolCallsFromMessage(msg)) {
				if (call.name === undefined) continue;
				const isRead = FILE_READ_TOOLS.has(call.name);
				const isModify = FILE_MODIFY_TOOLS.has(call.name);
				if (!isRead && !isModify) continue;
				// Find the path param from the message's raw call data.
				const paramsObj = this.extractToolCallParams(msg, call);
				let filePath: string | null = null;
				if (paramsObj) {
					for (const key of PATH_PARAM_KEYS) {
						const val = (paramsObj as Record<string, unknown>)[key];
						if (typeof val === 'string' && val.length > 0) { filePath = val; break; }
					}
				}
				if (!filePath) continue;
				(isModify ? filesModified : filesRead).add(filePath);
			}
		}
		return { filesRead: Array.from(filesRead), filesModified: Array.from(filesModified) };
	}

	/** Pull the params object for a specific tool call out of its message. */
	private extractToolCallParams(msg: LLMChatMessage, call: { id?: string; name: string }): Record<string, unknown> | null {
		if ('role' in msg && msg.role === 'assistant') {
			const m = msg as LLMChatMessage & { tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }>; content?: unknown };
			if (Array.isArray(m.tool_calls)) {
				for (const tc of m.tool_calls) {
					if (tc.function?.name === call.name && (!call.id || tc.id === call.id)) {
						try { return JSON.parse(tc.function.arguments ?? '{}'); } catch { return null; }
					}
				}
			}
			if (Array.isArray(m.content)) {
				for (const part of m.content as Array<{ type?: string; id?: string; name?: string; input?: Record<string, unknown> }>) {
					if (part?.type === 'tool_use' && part.name === call.name && (!call.id || part.id === call.id)) {
						return part.input ?? null;
					}
				}
			}
		}
		if ('parts' in msg && Array.isArray((msg as LLMChatMessage & { parts?: Array<{ functionCall?: { name?: string; args?: Record<string, unknown> } }> }).parts)) {
			for (const part of (msg as LLMChatMessage & { parts?: Array<{ functionCall?: { name?: string; args?: Record<string, unknown> } }> }).parts!) {
				if (part?.functionCall?.name === call.name) return part.functionCall.args ?? null;
			}
		}
		return null;
	}

	/**
	 * Compress messages to fit within target token limit using rolling window approach
	 *
	 * This method ensures we never exceed the context window by:
	 * - Preserving system message (if any)
	 * - Preserving the most recent N messages (recency bias)
	 * - Summarizing or removing older messages
	 */
	public async compressMessages(
		messages: LLMChatMessage[],
		modelName: string,
		config: Partial<CompressionConfig> = {}
	): Promise<{
		compressedMessages: LLMChatMessage[];
		stats: CompressionStats;
	}> {
		const fullConfig = { ...DEFAULT_COMPRESSION_CONFIG, ...config };
		const contextWindow = this.tokenCountingService.getContextWindowSize(modelName);

		// For large context windows (1M+ tokens), use conservative buffer for non-linear tokenization
		const largeContextBuffer = this.tokenCountingService.getLargeContextBuffer(contextWindow);
		const effectiveContextWindow = contextWindow - largeContextBuffer;

		// Calculate effective target (accounting for reserved tokens and large context buffer)
		const effectiveTarget = effectiveContextWindow - fullConfig.reservedTokens;
		const targetTokens = Math.floor(effectiveTarget * fullConfig.targetUsage);

		// Count tokens (use async for accuracy)
		let originalTokens: number;
		try {
			originalTokens = await this.tokenCountingService.countMessagesTokensAsync(messages, modelName);
		} catch (error) {
			// Fallback to character estimation — avoid JSON.stringify due to circular ref risk
			const safeStr = messages.map(m => {
				if ('content' in m && typeof m.content === 'string') return m.content
				if ('parts' in m && Array.isArray(m.parts)) return m.parts.map((p: any) => p.text || '').join(' ')
				return ''
			}).join(' ')
			originalTokens = this.estimateTokens(safeStr);
		}

		const stats: CompressionStats = {
			originalTokens,
			originalMessageCount: messages.length,
			targetTokens,
			finalTokens: 0,
			finalMessageCount: 0,
			messagesRemoved: 0,
			messagesSummarized: 0,
			toolResultsTruncated: 0,
			compressionRatio: 0,
		};

		// If already under target, no compression needed
		if (originalTokens <= targetTokens) {
			stats.finalTokens = originalTokens;
			stats.finalMessageCount = messages.length;
			stats.compressionRatio = 100;
			return { compressedMessages: messages, stats };
		}

		console.log(`[ContextCompression] Compressing ${messages.length} messages (${originalTokens} tokens) to target ${targetTokens} tokens (context: ${contextWindow})`);

		// Step 1: Identify system message, split messages, and identify critical messages
		const { systemMessage, recentMessages, oldMessages, criticalMessages } = this.splitMessages(messages, fullConfig.keepLastNMessages);

		console.log(`[ContextCompression] Identified ${criticalMessages.size} critical messages to preserve`);

		// Step 2: Truncate large tool results in all message categories
		const recentTrunc = this.truncateToolResults(recentMessages, fullConfig.maxToolResultLength);
		const oldTrunc = this.truncateToolResults(oldMessages, fullConfig.maxToolResultLength);
		let processedRecent = recentTrunc.messages;
		let processedOld = oldTrunc.messages;

		// Count truncated tool results (previously always reported 0 because
		// truncation never changes the message count, only message content).
		stats.toolResultsTruncated = recentTrunc.truncatedCount + oldTrunc.truncatedCount;

		// Step 3: Calculate target for recent + system (keep more for recency)
		const recentTargetTokens = Math.floor(targetTokens * 0.7); // 70% for recent messages
		const summaryTargetTokens = Math.floor(targetTokens * 0.25); // 25% for summary

		let currentTokens = await this.tokenCountingService.countMessagesTokensAsync(processedRecent, modelName);

		// If recent messages alone exceed target, we need emergency compression
		if (currentTokens > recentTargetTokens) {
			console.log(`[ContextCompression] Recent messages too large (${currentTokens} > ${recentTargetTokens}), applying emergency compression`);

			// Emergency: progressively reduce recent messages until fit
			processedRecent = await this.emergencyCompress(
				processedRecent,
				modelName,
				recentTargetTokens,
				fullConfig.emergencyKeepLastN
			);
			// Dropping messages mid-sequence can orphan tool calls/results — clean up.
			processedRecent = this.dropOrphanedToolMessages(processedRecent);
			currentTokens = await this.tokenCountingService.countMessagesTokensAsync(processedRecent, modelName);
		}

		// Step 4: Create summary of old messages if we have room
		let summaryMessage: LLMChatMessage | null = null;
		const availableForSummary = targetTokens - currentTokens;

		if (availableForSummary > 500 && processedOld.length > 0 && fullConfig.enableSummarization) {
			summaryMessage = this.createSummaryMessage(processedOld, Math.min(availableForSummary, summaryTargetTokens));
			stats.messagesSummarized = processedOld.length;
		} else if (processedOld.length > 0 && fullConfig.enableSummarization) {
			// No room for summary, but we still need to remove old messages
			stats.messagesRemoved = processedOld.length;
		}

		// Step 5: Combine messages (system + summary + recent)
		let finalMessages: LLMChatMessage[] = [];

		// Fold the summary into the system prompt where possible, instead of
		// emitting a separate mid-conversation "summary" message. A synthetic
		// user-role summary can be misread as a fresh user request, while a
		// synthetic system-role message mid-stream is rejected by separated-system
		// providers (Anthropic/Gemini keep the system message out of the array).
		// `systemMessage` is only present in-array for system-role/developer-role
		// providers, so folding into it is safe there; for separated providers we
		// fall back to a clearly-framed user message (the only provider-safe option).
		if (systemMessage) {
			const sysHasStringContent = 'content' in systemMessage && typeof systemMessage.content === 'string';
			if (summaryMessage && sysHasStringContent && 'content' in summaryMessage && typeof summaryMessage.content === 'string') {
				finalMessages.push({ ...systemMessage, content: `${systemMessage.content}\n\n${summaryMessage.content}` } as LLMChatMessage);
			} else {
				finalMessages.push(systemMessage);
				if (summaryMessage) {
					finalMessages.push(summaryMessage);
				}
			}
		} else if (summaryMessage) {
			finalMessages.push(summaryMessage);
		}

		finalMessages = finalMessages.concat(processedRecent);

		// Final safety check - if still over, emergency truncate recent messages
		let finalTokens = await this.tokenCountingService.countMessagesTokensAsync(finalMessages, modelName);
		if (finalTokens > targetTokens) {
			console.warn(`[ContextCompression] Still over target after compression (${finalTokens} > ${targetTokens}), applying emergency truncation`);
			finalMessages = await this.emergencyCompress(finalMessages, modelName, targetTokens, fullConfig.emergencyKeepLastN);
			// Dropping messages mid-sequence can orphan tool calls/results — clean up.
			finalMessages = this.dropOrphanedToolMessages(finalMessages);
			finalTokens = await this.tokenCountingService.countMessagesTokensAsync(finalMessages, modelName);
		}

		stats.finalTokens = finalTokens;
		stats.finalMessageCount = finalMessages.length;
		stats.compressionRatio = Math.round((finalTokens / originalTokens) * 100);

		console.log(`[ContextCompression] Result: ${stats.finalMessageCount} messages, ${stats.finalTokens} tokens (${stats.compressionRatio}% of original), removed ${stats.messagesRemoved}, summarized ${stats.messagesSummarized}`);

		return { compressedMessages: finalMessages, stats };
	}

	/**
	 * Split messages into system, recent, and old categories
	 * Also identifies critical user intent messages to preserve
	 */
	private splitMessages(messages: LLMChatMessage[], keepLastNMessages: number): {
		systemMessage: LLMChatMessage | null;
		recentMessages: LLMChatMessage[];
		oldMessages: LLMChatMessage[];
		criticalMessages: Set<number>; // Indices of messages to always preserve
	} {
		if (messages.length === 0) {
			return { systemMessage: null, recentMessages: [], oldMessages: [], criticalMessages: new Set() };
		}

		const criticalMessages = new Set<number>();

		// Check if first message is system message
		const firstMsg = messages[0];
		const hasSystemMessage = 'role' in firstMsg &&
			(firstMsg.role === 'system' || firstMsg.role === 'developer');

		// Always mark system message as critical
		if (hasSystemMessage) {
			criticalMessages.add(0);
		}

		// Mark recent user messages (especially the most recent user intent) as critical
		// Go backwards from the end to find the most recent user message
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			const role = ('role' in msg) ? msg.role : undefined;
			if (role === 'user') {
				criticalMessages.add(i);
				// Only mark the most recent user message as critical for intent preservation
				break;
			}
		}

		// Split so the last `keepLastNMessages` content messages are kept verbatim
		// as "recent" and everything before them is "old" (eligible for
		// compression/summarization). This respects the configured keepLastNMessages
		// instead of always splitting 50/50. If there's nothing older than the kept
		// window, oldMessages is empty (nothing to compress this pass).
		const splitPoint = (total: number) => Math.max(0, total - keepLastNMessages);

		if (hasSystemMessage) {
			// System message is first, then we have user/assistant/tool
			const systemMessage = firstMsg;
			const contentMessages = messages.slice(1);

			// Adjust critical indices for content messages (shift by 1)
			const adjustedCritical = new Set<number>();
			for (const idx of criticalMessages) {
				if (idx > 0) adjustedCritical.add(idx - 1);
			}

			const splitIdx = this.findValidCutPoint(contentMessages, splitPoint(contentMessages.length));
			const recentMessages = contentMessages.slice(splitIdx);
			const oldMessages = contentMessages.slice(0, splitIdx);

			return { systemMessage, recentMessages, oldMessages, criticalMessages: adjustedCritical };
		} else {
			// No system message, all messages are content
			const splitIdx = this.findValidCutPoint(messages, splitPoint(messages.length));
			const recentMessages = messages.slice(splitIdx);
			const oldMessages = messages.slice(0, splitIdx);

			return { systemMessage: null, recentMessages, oldMessages, criticalMessages };
		}
	}

	/**
	 * Emergency compression: progressively reduce messages until they fit
	 * Preserves critical messages (system, most recent user intent)
	 */
	private async emergencyCompress(
		messages: LLMChatMessage[],
		modelName: string,
		maxTokens: number,
		minKeepMessages: number
	): Promise<LLMChatMessage[]> {
		if (messages.length <= minKeepMessages) {
			return messages;
		}

		// Identify critical messages that should never be removed
		const criticalIndices = new Set<number>();

		// Check if first message is system/developer message
		const firstMsg = messages[0];
		if ('role' in firstMsg && (firstMsg.role === 'system' || firstMsg.role === 'developer')) {
			criticalIndices.add(0);
		}

		// Find the most recent user message (preserves user intent)
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			const role = ('role' in msg) ? msg.role : undefined;
			if (role === 'user') {
				criticalIndices.add(i);
				break; // Only preserve most recent user message
			}
		}

		let currentMessages = [...messages];
		let iterations = 0;
		const maxIterations = 30; // Prevent infinite loops

		while (iterations < maxIterations) {
			const currentTokens = await this.tokenCountingService.countMessagesTokensAsync(currentMessages, modelName);

			if (currentTokens <= maxTokens) {
				break;
			}

			// Early break if we've already reduced to minimum
			if (currentMessages.length <= minKeepMessages) {
				break;
			}

			// Recompute critical indices based on the CURRENT array state
			const currentCriticalIndices = new Set<number>();
			const currentFirstMsg = currentMessages[0];
			if ('role' in currentFirstMsg && (currentFirstMsg.role === 'system' || currentFirstMsg.role === 'developer')) {
				currentCriticalIndices.add(0);
			}
			for (let i = currentMessages.length - 1; i >= 0; i--) {
				const msg = currentMessages[i];
				const role = ('role' in msg) ? msg.role : undefined;
				if (role === 'user') {
					currentCriticalIndices.add(i);
					break;
				}
			}

			// Drop the oldest non-critical messages, but ALWAYS keep every critical
			// message (system/developer prompt + most recent user message). A previous
			// version used a single `slice(finalKeepStart)` from the start, which
			// silently dropped the critical prefix — including the system message —
			// whenever the kept window started past it. Filtering by index preserves
			// critical messages in place while shedding the oldest non-critical ones.
			const nonCriticalIndices: number[] = [];
			for (let i = 0; i < currentMessages.length; i++) {
				if (!currentCriticalIndices.has(i)) {
					nonCriticalIndices.push(i);
				}
			}
			const keepNonCriticalCount = Math.floor(nonCriticalIndices.length * 0.7);
			const keepNonCritical = new Set(nonCriticalIndices.slice(-keepNonCriticalCount));
			currentMessages = currentMessages.filter((_, i) => currentCriticalIndices.has(i) || keepNonCritical.has(i));
			iterations++;
		}

		if (iterations >= maxIterations) {
			console.warn(`[ContextCompression] Emergency compression hit max iterations, preserving critical messages`);
			// Recompute critical indices one final time from the current state
			const finalCriticalIndices = new Set<number>();
			const finalFirstMsg = currentMessages[0];
			if ('role' in finalFirstMsg && (finalFirstMsg.role === 'system' || finalFirstMsg.role === 'developer')) {
				finalCriticalIndices.add(0);
			}
			for (let i = currentMessages.length - 1; i >= 0; i--) {
				const msg = currentMessages[i];
				const role = ('role' in msg) ? msg.role : undefined;
				if (role === 'user') {
					finalCriticalIndices.add(i);
					break;
				}
			}

			// At minimum, return critical messages if available
			if (finalCriticalIndices.size > 0) {
				const firstCriticalIdx = finalCriticalIndices.values().next().value;
				if (firstCriticalIdx !== undefined && finalCriticalIndices.size === 1) {
					return [currentMessages[firstCriticalIdx]];
				}
				return currentMessages.filter((_, i) => finalCriticalIndices.has(i));
			}
			return currentMessages.slice(-minKeepMessages);
		}

		// Dropping messages mid-sequence can orphan tool calls/results — clean up.
		currentMessages = this.dropOrphanedToolMessages(currentMessages);
		return currentMessages;
	}

	/**
	 * Create a summary message from old messages
	 */
	private createSummaryMessage(oldMessages: LLMChatMessage[], maxTokens: number): LLMChatMessage {
		// Extract key information from old messages
		const summaryParts: string[] = [];
		let totalLength = 0;

		for (const msg of oldMessages) {
			// Extract content based on message format
			let content = '';

			// Handle string content
			if ('content' in msg && typeof msg.content === 'string') {
				content = msg.content;
			}
			// Handle array content (OpenAI/Anthropic format)
			else if ('content' in msg && Array.isArray(msg.content)) {
				content = msg.content
					.map((part: any) => {
						if ('text' in part) return part.text;
						if ('thinking' in part) return `[thinking: ${part.thinking.substring(0, 100)}...]`;
						if ('tool_result' in part) return `[result for ${(msg as any).name || 'tool'}]`;
						return '';
					})
					.join(' | ');
			}
			// Handle parts format (Gemini)
			else if ('parts' in msg) {
				content = msg.parts
					.map((part: any) => {
						if ('text' in part) return part.text;
						if ('functionCall' in part) return `[called ${part.functionCall.name}]`;
						if ('functionResponse' in part) return `[response from ${part.functionResponse.name}]`;
						return '';
					})
					.join(' | ');
			}

			// Truncate each message to a preview
			const preview = content.split(/\n\n|\n/).slice(0, 2).join(' ').substring(0, 500);
			if (preview.length > 10) {
				const role = (msg as any).role || 'msg';
				const entry = `[${role}]: ${preview}`;

				if (totalLength + entry.length < maxTokens * 4) { // Rough char estimate
					summaryParts.push(entry);
					totalLength += entry.length;
				} else {
					break;
				}
			}
		}

		// FILE-OP TRACKING (ported from a-coder-cli compaction): preserve which
		// files were read and modified in the summarized region so the agent keeps
		// its workspace bearings after compression.
		const { filesRead, filesModified } = this.extractFileOperations(oldMessages);
		const fileOpLines: string[] = [];
		if (filesModified.length > 0) {
			fileOpLines.push(`FILES MODIFIED: ${filesModified.join(', ')}`);
		}
		if (filesRead.length > 0) {
			fileOpLines.push(`FILES READ: ${filesRead.join(', ')}`);
		}

		const summaryText = summaryParts.length > 0
			? `[PREVIOUS CONVERSATION SUMMARY - ${oldMessages.length} messages condensed. This is background context from earlier in the conversation, NOT a new request from the user. Do not respond to it directly.]${fileOpLines.length > 0 ? `\n\n${fileOpLines.join('\n')}` : ''}\n\n${summaryParts.join('\n')}\n\n[End of summary]`
			: `[Previous conversation context condensed. This is background context, NOT a new request from the user.]${fileOpLines.length > 0 ? ` ${fileOpLines.join('. ')}` : ''}`;

		// Return as a user-role message. This is only used as a fallback for
		// separated-system providers (Anthropic/Gemini) where the system message
		// is not part of the message array; for system-role providers the summary
		// is folded into the system prompt by the caller instead. A user role is
		// the only mid-stream role those APIs accept, and the framing above makes
		// clear it is context, not a fresh user instruction.
		return {
			role: 'user',
			content: summaryText
		} as LLMChatMessage;
	}

	/**
	 * Fallback token estimation when IPC fails
	 */
	private estimateTokens(text: string): number {
		// More accurate estimation: ~4 chars per token for English, but varies
		return Math.ceil(text.length / 4) + 10;
	}

	/**
	 * Smart truncate JSON string to avoid context overflow
	 */
	private truncateJsonString(jsonStr: string, maxLength: number): string {
		if (jsonStr.length <= maxLength) return jsonStr;

		try {
			const obj = JSON.parse(jsonStr);
			let modified = false;

			const truncateRecursive = (o: any) => {
				if (!o || typeof o !== 'object') return;

				for (const key of Object.keys(o)) {
					if (typeof o[key] === 'string' && o[key].length > maxLength) {
						o[key] = o[key].substring(0, maxLength) + `\n... [truncated ${o[key].length - maxLength} chars]`;
						modified = true;
					} else if (typeof o[key] === 'object') {
						truncateRecursive(o[key]);
					}
				}
			};

			truncateRecursive(obj);

			if (modified) {
				return JSON.stringify(obj);
			}
		} catch (e) {
			// If invalid JSON, fallback to simple truncation if significantly larger
			if (jsonStr.length > maxLength * 2) {
				return jsonStr.substring(0, maxLength * 2) + `\n... [truncated raw string]`;
			}
		}
		return jsonStr;
	}

	/**
	 * Truncate large tool results and tool calls to reduce token usage.
	 * Returns the (possibly modified) messages plus a count of how many individual
	 * tool results/calls were truncated, for accurate stats.
	 */
	private truncateToolResults(messages: LLMChatMessage[], maxLength: number): { messages: LLMChatMessage[]; truncatedCount: number } {
		let truncatedCount = 0
		const result = messages.map(msg => {
			// OpenAI tool format (role: tool)
			if ('role' in msg && msg.role === 'tool' && 'content' in msg) {
				if (typeof msg.content === 'string' && msg.content.length > maxLength) {
					truncatedCount++
					return {
						...msg,
						content: msg.content.substring(0, maxLength) + `\n\n[... truncated ${msg.content.length - maxLength} characters for context window management]`
					};
				}
			}

			// OpenAI tool calls (role: assistant)
			if ('role' in msg && msg.role === 'assistant' && 'tool_calls' in msg && msg.tool_calls) {
				let modified = false
				const newToolCalls = msg.tool_calls.map(tc => {
					if (tc.function.arguments.length > maxLength) {
						truncatedCount++
						modified = true
						return {
							...tc,
							function: {
								...tc.function,
								arguments: this.truncateJsonString(tc.function.arguments, maxLength)
							}
						};
					}
					return tc;
				});
				if (modified) {
					return { ...msg, tool_calls: newToolCalls } as LLMChatMessage;
				}
			}

			// Anthropic format
			if ('content' in msg && Array.isArray(msg.content)) {
				let modified = false
				const newContent = msg.content.map(part => {
					// Tool result. `content` may be a string OR an array of content
					// blocks (Anthropic allows both); only truncate the string form,
					// otherwise `.substring` would throw at runtime.
					if ('type' in part && part.type === 'tool_result' && 'content' in part && typeof part.content === 'string') {
						if (part.content.length > maxLength) {
							truncatedCount++
							modified = true
							return {
								...part,
								content: part.content.substring(0, maxLength) + `\n\n[... truncated for context window]`
							};
						}
					}
					// Tool use (assistant)
					if ('type' in part && part.type === 'tool_use' && 'input' in part) {
						const newInput = { ...part.input };
						for (const key of Object.keys(newInput)) {
							if (typeof newInput[key] === 'string' && newInput[key].length > maxLength) {
								newInput[key] = newInput[key].substring(0, maxLength) + `\n... [truncated]`;
								modified = true;
								truncatedCount++
							}
						}
						if (modified) {
							return { ...part, input: newInput };
						}
					}
					return part;
				});
				if (modified) {
					return { ...msg, content: newContent } as LLMChatMessage;
				}
			}

			// Gemini format
			if ('parts' in msg) {
				let modified = false
				const newParts = msg.parts.map(part => {
					// Function response. `output` is typically a string but may be an
					// object for structured responses; only truncate the string form.
					if ('functionResponse' in part) {
						const output = part.functionResponse.response.output;
						if (typeof output === 'string' && output.length > maxLength) {
							truncatedCount++
							modified = true
							return {
								...part,
								functionResponse: {
									...part.functionResponse,
									response: {
										output: output.substring(0, maxLength) + `\n\n[... truncated for context window]`
									}
								}
							};
						}
					}
					// Function call
					if ('functionCall' in part) {
						const newArgs = { ...part.functionCall.args };
						for (const key of Object.keys(newArgs)) {
							if (typeof newArgs[key] === 'string' && (newArgs[key] as string).length > maxLength) {
								newArgs[key] = (newArgs[key] as string).substring(0, maxLength) + `\n... [truncated]`;
								modified = true;
								truncatedCount++
							}
						}
						if (modified) {
							return {
								...part,
								functionCall: { ...part.functionCall, args: newArgs }
							};
						}
					}
					return part;
				});
				if (modified) {
					return { ...msg, parts: newParts } as LLMChatMessage;
				}
			}

			return msg;
		});
		return { messages: result, truncatedCount };
	}

	/**
	 * Check if compression is needed (async version for accuracy)
	 */
	public async needsCompression(
		messages: LLMChatMessage[],
		modelName: string,
		threshold: number = 0.8
	): Promise<boolean> {
		const currentTokens = await this.tokenCountingService.countMessagesTokensAsync(messages, modelName);
		const contextWindow = this.tokenCountingService.getContextWindowSize(modelName);
		const usage = currentTokens / contextWindow;

		return usage > threshold;
	}

	/**
	 * Get compression statistics without actually compressing (async version for accuracy)
	 */
	public async getCompressionPreview(
		messages: LLMChatMessage[],
		modelName: string,
		config: Partial<CompressionConfig> = {}
	): Promise<CompressionStats> {
		const { stats } = await this.compressMessages(messages, modelName, config);
		return stats;
	}
}
