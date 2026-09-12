/*--------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { LLMChatMessage } from './sendLLMMessageTypes.js';
import { IMainProcessService } from '../../../../platform/ipc/common/mainProcessService.js';
import { CompressionConfig, CompressionStats } from './contextCompressionService.js';
import { IVoidSettingsService } from './voidSettingsService.js';
import { ProviderName, providerNames } from './voidSettingsTypes.js';
import { Disposable, toDisposable } from '../../../../base/common/lifecycle.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { voidDevLog, voidDevWarn } from './devLog.js';
import { registerSingleton, InstantiationType } from '../../../../platform/instantiation/common/extensions.js';

export interface ITokenCountingService {
	readonly _serviceBrand: undefined;
	countTextTokens(text: string, modelName: string): number;
	countMessageTokens(message: LLMChatMessage, modelName: string): number;
	countMessagesTokens(messages: LLMChatMessage[], modelName: string): number;
	countTextTokensAsync(text: string, modelName: string): Promise<number>;
	countMessageTokensAsync(message: LLMChatMessage, modelName: string): Promise<number>;
	countMessagesTokensAsync(messages: LLMChatMessage[], modelName: string): Promise<number>;
	/**
	 * Synchronous chars/4 estimation of message tokens — no IPC, no ratio
	 * correction. Used for cheap estimates of small trailing deltas on top of a
	 * real usage anchor; NOT a substitute for a full accurate count.
	 */
	estimateMessagesTokensFast(messages: ReadonlyArray<{ content: string | ReadonlyArray<string | { text?: string }> }>): number;
	getContextWindowSize(modelName: string): number;
	getRemainingTokens(messages: LLMChatMessage[], modelName: string): number;
	getRemainingTokensAsync(messages: LLMChatMessage[], modelName: string): Promise<number>;
	fitsInContextWindow(messages: LLMChatMessage[], modelName: string): boolean;
	fitsInContextWindowAsync(messages: LLMChatMessage[], modelName: string): Promise<boolean>;
	needsCompressionAsync(messages: LLMChatMessage[], modelName: string, threshold?: number): Promise<boolean>;
	getCompressionPreviewAsync(messages: LLMChatMessage[], modelName: string, config?: Partial<CompressionConfig>): Promise<CompressionStats>;
	getLargeContextBuffer(contextWindow: number): number;
	getEffectiveContextWindow(modelName: string): number;
	setDynamicContextWindow(modelName: string, contextWindow: number): void;
	updateTokenRatio(modelName: string, estimatedTokens: number, actualTokens: number): void;
	clearExpiredCaches(): void;
}

export const ITokenCountingService = createDecorator<ITokenCountingService>('tokenCountingService');

/**
 * Service for counting tokens in messages and managing context windows.
 * Uses tiktoken via IPC; falls back to character estimation if IPC fails.
 */
export class TokenCountingService extends Disposable implements ITokenCountingService {
	readonly _serviceBrand: undefined;
	private readonly _modelRatios = new Map<string, number>();

	// MEMORY OPTIMIZATION: Cache for recent counts to avoid redundant IPC
	private _countCache = new Map<string, number>();
	private readonly MAX_CACHE_SIZE = 50;

	// PERFORMANCE: Cache for partial message histories to avoid redundant IPC
	private _historyCache = new Map<string, { count: number, lastMessageHash: string }>();
	private readonly MAX_HISTORY_CACHE_SIZE = 20;

	// MEMORY OPTIMIZATION: TTL for cache entries (10 minutes)
	private _cacheTimestamps = new Map<string, number>();
	private readonly CACHE_TTL_MS = 10 * 60 * 1000;

	// CRITICAL OPERATIONS: Track pending async operations to avoid duplicate IPC calls
	private _pendingAsyncCounts = new Map<string, Promise<number>>();

	// DYNAMIC CONTEXT WINDOWS: Cache for dynamically fetched context windows
	private _dynamicContextWindows = new Map<string, { window: number; timestamp: number }>();
	private readonly CONTEXT_WINDOW_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

	constructor(
		@IMainProcessService private readonly mainProcessService: IMainProcessService,
		@IVoidSettingsService private readonly voidSettingsService: IVoidSettingsService,
	) {
		super();
		voidDevLog('[TokenCountingService] Using tiktoken via IPC with character-based fallback');
		// Start cache cleanup interval
		const interval = setInterval(() => this._cleanupExpiredCache(), this.CACHE_TTL_MS / 2);
		this._register(toDisposable(() => clearInterval(interval)));
	}

	/**
	 * Clean up expired cache entries
	 */
	private _cleanupExpiredCache(): void {
		const now = Date.now();
		for (const [key, timestamp] of this._cacheTimestamps.entries()) {
			if (now - timestamp > this.CACHE_TTL_MS) {
				this._cacheTimestamps.delete(key);
				this._countCache.delete(key);
			}
		}
		// Clean up history cache separately
		for (const [hKey] of this._historyCache.entries()) {
			// History cache uses TTL from main cache entries
			const modelName = hKey.split(':')[0];
			for (const [tsKey, tsValue] of this._cacheTimestamps.entries()) {
				if (tsKey.startsWith(modelName) && now - tsValue > this.CACHE_TTL_MS) {
					this._historyCache.delete(hKey);
					break;
				}
			}
		}
	}


	/**
	 * Update the token ratio for a specific model based on actual usage.
	 * Helps improve character-based estimation for future calls.
	 */
	public updateTokenRatio(modelName: string, estimatedTokens: number, actualTokens: number): void {
		if (estimatedTokens <= 0 || actualTokens <= 0) return;

		const ratio = actualTokens / estimatedTokens;

		// Get existing ratio or default to the baseline multiplier
		const existingRatio = this._modelRatios.get(modelName) || this._getTokenCountMultiplier(modelName);

		// Smoothing factor to avoid wild fluctuations (EMA - Exponential Moving Average)
		// New ratio = 0.7 * old + 0.3 * new
		const smoothedRatio = (existingRatio * 0.7) + (ratio * 0.3);

		// Cap the ratio to reasonable bounds (0.5 to 5.0)
		const cappedRatio = Math.max(0.5, Math.min(5.0, smoothedRatio));

		this._modelRatios.set(modelName, cappedRatio);
		voidDevLog(`[TokenCountingService] Updated token ratio for ${modelName}: ${cappedRatio.toFixed(3)} (last: ${ratio.toFixed(3)})`);
	}

	/**
	 * Count tokens in a single text string
	 * Synchronous version returns estimate to avoid blocking UI.
	 */
	public countTextTokens(text: string, modelName: string): number {
		// MEMORY OPTIMIZATION: Return estimate immediately without triggering fire-and-forget IPC.
		// Frequent synchronous calls during typing or rendering shouldn't flood the IPC channel.
		const multiplier = this._getTokenCountMultiplier(modelName);
		return Math.ceil((text.length / 4) * multiplier);
	}

	/**
	 * Count tokens in a chat message
	 * Synchronous version returns estimate to avoid blocking UI.
	 */
	public countMessageTokens(message: LLMChatMessage, modelName: string): number {
		const multiplier = this._getTokenCountMultiplier(modelName);
		return Math.ceil(this._estimateSingleMessageTokens(message) * multiplier);
	}

	/**
	 * Count tokens in an array of chat messages
	 * Synchronous version returns estimate to avoid blocking UI.
	 */
	public countMessagesTokens(messages: LLMChatMessage[], modelName: string): number {
		const multiplier = this._getTokenCountMultiplier(modelName);
		let totalTokens = 0;
		for (const message of messages) {
			totalTokens += this._estimateSingleMessageTokens(message);
		}
		totalTokens += 3;
		return Math.ceil(totalTokens * multiplier);
	}

	/**
	 * Get token count multiplier for models with non-standard tokenizers
	 * Minimax and some other models have very different tokenization than cl100k_base
	 */
	private _getTokenCountMultiplier(modelName: string): number {
		// If we have a dynamically calculated ratio, use it
		if (this._modelRatios.has(modelName)) {
			return this._modelRatios.get(modelName)!;
		}

		const lowerName = modelName.toLowerCase();
		if (lowerName.includes('minimax')) {
			return 1.7; // Empirical observation: 128k estimated -> >204k actual
		}
		return 1.0;
	}

	/**
	 * Async version: count tokens in a single text string using tiktoken via IPC.
	 * Falls back to character estimation on IPC error.
	 * Deduplicates concurrent calls for the same text.
	 */
	public async countTextTokensAsync(text: string, modelName: string): Promise<number> {
		const cacheKey = `${modelName}:${text.length}:${text.substring(0, 50)}`;

		// Check cache with TTL
		const cached = this._getCachedWithTTL(text, modelName);
		if (cached !== undefined) {
			return cached;
		}

		// Deduplicate concurrent calls
		const pendingKey = `text:${cacheKey}`;
		if (this._pendingAsyncCounts.has(pendingKey)) {
			return this._pendingAsyncCounts.get(pendingKey)!;
		}

		const countPromise = this._countTextTokensAsyncInternal(text, modelName);
		this._pendingAsyncCounts.set(pendingKey, countPromise);

		try {
			const result = await countPromise;
			return result;
		} finally {
			this._pendingAsyncCounts.delete(pendingKey);
		}
	}

	private async _countTextTokensAsyncInternal(text: string, modelName: string): Promise<number> {
		const multiplier = this._getTokenCountMultiplier(modelName);

		try {
			const channel = this.mainProcessService.getChannel('void-channel-token-counting');
			const count = await channel.call('countTokens', { text, modelName });
			const finalCount = Math.ceil((typeof count === 'number' ? count : Math.ceil(text.length / 4)) * multiplier);
			this._setCachedWithTimestamp(text, modelName, finalCount);
			return finalCount;
		} catch (error) {
			voidDevWarn('[TokenCountingService] IPC token counting failed, using character estimate:', error);
			return Math.ceil((text.length / 4) * multiplier);
		}
	}

	/**
	 * Check cache with TTL validation
	 */
	private _getCachedWithTTL(text: string, modelName: string): number | undefined {
		const cacheKey = `${modelName}:${text.length}:${text.substring(0, 50)}`;
		const timestamp = this._cacheTimestamps.get(cacheKey);
		if (timestamp === undefined) return undefined;

		if (Date.now() - timestamp > this.CACHE_TTL_MS) {
			this._cacheTimestamps.delete(cacheKey);
			this._countCache.delete(cacheKey);
			return undefined;
		}

		return this._countCache.get(cacheKey);
	}

	private _setCachedWithTimestamp(text: string, modelName: string, count: number): void {
		const cacheKey = `${modelName}:${text.length}:${text.substring(0, 50)}`;

		if (this._countCache.size >= this.MAX_CACHE_SIZE) {
			const firstKey = this._countCache.keys().next().value;
			if (firstKey) {
				this._countCache.delete(firstKey);
				this._cacheTimestamps.delete(firstKey);
			}
		}

		this._countCache.set(cacheKey, count);
		this._cacheTimestamps.set(cacheKey, Date.now());
	}

	/**
	 * Helper to extract content from different message formats
	 */
	private _extractContent(message: LLMChatMessage): string {
		let contentStr = '';

		// OpenAI/Anthropic format
		if ('content' in message) {
			if (typeof message.content === 'string') {
				contentStr += message.content;
			}
			// Handle array content (Anthropic format with reasoning/multi-part)
			else if (Array.isArray(message.content)) {
				contentStr += message.content
					.map(part => {
						if (typeof part === 'string') return part;
						if ('text' in part) return part.text;
						if ('thinking' in part) return `[thinking]${part.thinking}[/thinking]`;
						if ('type' in part && part.type === 'tool_use') return `[tool_use:${part.name}:${JSON.stringify(part.input)}]`;
						if ('type' in part && part.type === 'tool_result') return `[tool_result:${part.content}]`;
						return '';
					})
					.join('');
			}
		}

		// OpenAI tool calls (separate property)
		if ('tool_calls' in message && Array.isArray(message.tool_calls)) {
			contentStr += message.tool_calls.map(tc =>
				`[tool_call:${tc.function.name}:${tc.function.arguments}]`
			).join('');
		}

		// Gemini format
		if ('parts' in message) {
			return message.parts
				.map(part => {
					if ('text' in part) return part.text;
					if ('functionCall' in part) return `[function_call:${part.functionCall.name}:${JSON.stringify(part.functionCall.args)}]`;
					if ('functionResponse' in part) return `[function_response:${part.functionResponse.name}:${JSON.stringify(part.functionResponse.response)}]`;
					return '';
				})
				.join('');
		}

		return contentStr;
	}

	/**
	 * Helper to extract role from different message formats
	 */
	private _extractRole(message: LLMChatMessage): string {
		if ('role' in message) {
			return message.role;
		}
		return 'unknown';
	}

	/**
	 * Async version: count tokens in a single chat message using tiktoken via IPC.
	 * Falls back to character estimation on IPC error.
	 */
	public async countMessageTokensAsync(message: LLMChatMessage, modelName: string): Promise<number> {
		const multiplier = this._getTokenCountMultiplier(modelName);
		try {
			const channel = this.mainProcessService.getChannel('void-channel-token-counting');
			const plainMessage = {
				role: this._extractRole(message),
				content: this._extractContent(message)
			};
			const count = await channel.call('countMessagesTokens', { messages: [plainMessage], modelName });
			const baseCount = typeof count === 'number' ? count : Math.ceil(JSON.stringify(message).length / 4) + 4;
			return Math.ceil(baseCount * multiplier);
		} catch (error) {
			voidDevWarn('[TokenCountingService] IPC token counting failed, using character estimate:', error);
			const messageStr = JSON.stringify(message);
			return Math.ceil(((messageStr.length / 4) + 4) * multiplier);
		}
	}

	/**
	 * Async version: count tokens in an array of chat messages using tiktoken via IPC.
	 * Falls back to character estimation on IPC error.
	 * Deduplicates concurrent calls and uses TTL-based caching.
	 * CRITICAL: Use this for compression decisions and context window checks.
	 */
	public async countMessagesTokensAsync(messages: LLMChatMessage[], modelName: string): Promise<number> {
		if (messages.length === 0) return 0;

		// Generate cache key based on content hashes. NOTE: the modelName prefix
		// is added once inside _getCachedWithTTLByKey / _setCachedWithTTLByKey, so
		// the key here must NOT already include it (a previous version doubled the
		// prefix, so reads never matched writes).
		const contentHash = this._hashMessages(messages);
		const cacheKey = `${messages.length}:${contentHash}`;

		// Check cache with TTL
		const cached = this._getCachedWithTTLByKey(cacheKey, modelName);
		if (cached !== undefined) {
			return cached;
		}

		// Deduplicate concurrent calls
		const pendingKey = `messages:${cacheKey}`;
		if (this._pendingAsyncCounts.has(pendingKey)) {
			return this._pendingAsyncCounts.get(pendingKey)!;
		}

		const countPromise = this._countMessagesTokensAsyncInternal(messages, modelName);
		this._pendingAsyncCounts.set(pendingKey, countPromise);

		try {
			const result = await countPromise;
			// Persist under the same key the read uses (modelName:length:hash).
			this._setCachedWithTTLByKey(cacheKey, modelName, result);
			return result;
		} finally {
			this._pendingAsyncCounts.delete(pendingKey);
		}
	}

	private async _countMessagesTokensAsyncInternal(messages: LLMChatMessage[], modelName: string): Promise<number> {
		const multiplier = this._getTokenCountMultiplier(modelName);

		// PERFORMANCE: Check history cache for O(1) count if only last message changed
		if (messages.length > 1) {
			const historyKey = `${modelName}:${messages.length}`;
			const cached = this._historyCache.get(historyKey);
			const lastMsg = messages[messages.length - 1];
			const lastMsgContent = this._extractContent(lastMsg);
			const lastMsgHash = `${lastMsgContent.length}:${lastMsgContent.substring(0, 50)}`;

			if (cached && cached.lastMessageHash === lastMsgHash) {
				return cached.count;
			}
		}

		try {
			const channel = this.mainProcessService.getChannel('void-channel-token-counting');
			const plainMessages = messages.map(msg => ({
				role: this._extractRole(msg),
				content: this._extractContent(msg)
			}));
			const count = await channel.call('countMessagesTokens', { messages: plainMessages, modelName });
			const baseCount = typeof count === 'number' ? count : this._estimateMessagesTokens(messages);
			const finalCount = Math.ceil(baseCount * multiplier);

			// Update history cache
			if (messages.length > 1) {
				const historyKey = `${modelName}:${messages.length}`;
				const lastMsg = messages[messages.length - 1];
				const lastMsgContent = this._extractContent(lastMsg);
				const lastMsgHash = `${lastMsgContent.length}:${lastMsgContent.substring(0, 50)}`;

				if (this._historyCache.size >= this.MAX_HISTORY_CACHE_SIZE) {
					const firstKey = this._historyCache.keys().next().value;
					if (firstKey) this._historyCache.delete(firstKey);
				}
				this._historyCache.set(historyKey, { count: finalCount, lastMessageHash: lastMsgHash });
			}

			return finalCount;
		} catch (error) {
			voidDevWarn('[TokenCountingService] IPC token counting failed, using character estimate:', error);
			return Math.ceil(this._estimateMessagesTokens(messages) * multiplier);
		}
	}

	/**
	 * Hash messages for cache key generation
	 */
	private _hashMessages(messages: LLMChatMessage[]): string {
		// Simple hash based on message count and content lengths
		// More sophisticated hashing would be overkill for cache deduplication
		let hash = messages.length.toString();
		for (let i = 0; i < messages.length; i++) {
			const msg = messages[i];
			const content = this._extractContent(msg);
			hash += `:${i}:${content.length}:${content.substring(0, 20)}`;
		}
		return hash;
	}

	/**
	 * Check cache with TTL validation using pre-computed cacheKey
	 */
	private _getCachedWithTTLByKey(cacheKey: string, modelName: string): number | undefined {
		const fullKey = `${modelName}:${cacheKey}`;
		const timestamp = this._cacheTimestamps.get(fullKey);
		if (timestamp === undefined) return undefined;

		if (Date.now() - timestamp > this.CACHE_TTL_MS) {
			this._cacheTimestamps.delete(fullKey);
			this._countCache.delete(fullKey);
			return undefined;
		}

		return this._countCache.get(fullKey);
	}

	/**
	 * Store a count in the TTL cache using the same pre-computed cacheKey the
	 * read path uses (modelName prefix added once here).
	 */
	private _setCachedWithTTLByKey(cacheKey: string, modelName: string, count: number): void {
		const fullKey = `${modelName}:${cacheKey}`;
		if (this._countCache.size >= this.MAX_CACHE_SIZE) {
			const firstKey = this._countCache.keys().next().value;
			if (firstKey) {
				this._countCache.delete(firstKey);
				this._cacheTimestamps.delete(firstKey);
			}
		}
		this._countCache.set(fullKey, count);
		this._cacheTimestamps.set(fullKey, Date.now());
	}


	/**
	 * Fast synchronous estimate for small message deltas (see interface doc).
	 * Follows the file-wide CHARS_PER_TOKEN = 4 convention.
	 */
	public estimateMessagesTokensFast(messages: ReadonlyArray<{ content: string | ReadonlyArray<string | { text?: string }> }>): number {
		let chars = 0;
		for (const message of messages) {
			if (typeof message.content === 'string') {
				chars += message.content.length;
			} else if (Array.isArray(message.content)) {
				for (const part of message.content) {
					if (typeof part === 'string') chars += part.length;
					else chars += part?.text?.length ?? 0;
				}
			}
		}
		// ~4 chars/token plus a small per-message and per-request overhead
		return Math.ceil(chars / 4) + messages.length * 4 + 3;
	}

	/**
	 * Internal helper: character-based estimate for messages (fallback).
	 * Optimized to avoid full JSON.stringify which is slow for large objects.
	 */
	private _estimateMessagesTokens(messages: LLMChatMessage[]): number {
		let totalTokens = 0;
		for (const message of messages) {
			totalTokens += this._estimateSingleMessageTokens(message);
		}
		totalTokens += 3;
		return totalTokens;
	}

	/**
	 * Estimate tokens for a single message without full JSON serialization.
	 */
	private _estimateSingleMessageTokens(message: LLMChatMessage): number {
		let contentLen = 0;

		// Role is short (user/assistant/system)
		const roleLen = this._extractRole(message).length;

		// Content is the main part
		if ('content' in message) {
			if (typeof message.content === 'string') {
				contentLen = message.content.length;
			} else if (Array.isArray(message.content)) {
				for (const part of message.content) {
					if (typeof part === 'string') contentLen += (part as string).length;
					else if ('text' in part) contentLen += (part as any).text?.length || 0;
					else if ('thinking' in part) contentLen += (part as any).thinking?.length || 0;
					else if ('type' in part && ((part as any).type === 'tool_use' || (part as any).type === 'tool_result')) {
						// Small overhead for tool metadata + serialized input/content
						const p = part as any;
						contentLen += (p.name?.length || 0) + (JSON.stringify(p.input || p.content || '').length);
					}
				}
			}
		} else if ('parts' in message && Array.isArray(message.parts)) {
			for (const part of message.parts) {
				if ('text' in part) contentLen += part.text?.length || 0;
				else contentLen += JSON.stringify(part).length; // Fallback for complex parts
			}
		}

		// Tool calls (OpenAI format)
		if ('tool_calls' in message && Array.isArray(message.tool_calls)) {
			for (const tc of message.tool_calls) {
				contentLen += (tc.function?.name?.length || 0) + (tc.function?.arguments?.length || 0) + 10;
			}
		}

		return Math.ceil((roleLen + contentLen + 20) / 4) + 4;
	}


	/**
	 * Parse a full model identifier into its provider and model name parts.
	 * Handles "provider:model" and "provider:org/model:tag" formats.
	 */
	private _parseProviderAndModel(modelName: string): { providerName?: ProviderName; modelName: string } {
		const parts = modelName.split(':');
		let providerName: string | undefined;
		let cleanName = modelName;

		if (parts.length > 2) {
			providerName = parts[0];
			cleanName = parts.slice(1).join(':');
		} else if (parts.length === 2) {
			providerName = parts[0];
			cleanName = parts[1];
		}

		if (providerName && providerNames.includes(providerName as ProviderName)) {
			return { providerName: providerName as ProviderName, modelName: cleanName };
		}

		return { modelName: cleanName };
	}

	/**
	 * Get the context window size for a model.
	 * Checks user overrides first, then dynamic cache, then falls back to static mappings.
	 * For models with unknown context windows, attempts to fetch dynamically.
	 */
	public getContextWindowSize(modelName: string): number {
		const { providerName, modelName: cleanModelName } = this._parseProviderAndModel(modelName);

		// Respect user overrides from the Model Management page first
		if (providerName) {
			const override = this.voidSettingsService.state.overridesOfModel[providerName]?.[cleanModelName]?.contextWindow;
			if (typeof override === 'number' && override > 0) {
				return override;
			}
		}

		// Check dynamic cache next
		const cached = this._getDynamicContextWindow(modelName);
		if (cached !== undefined) {
			return cached;
		}

		// Use the model name already stripped of provider prefix by _parseProviderAndModel.
		// Handle OpenRouter "org/model" format (e.g., "x-ai/grok-4.1-fast" → "grok-4.1-fast")
		let cleanName = cleanModelName;
		if (cleanName.includes('/')) {
			cleanName = cleanName.split('/').pop() || cleanName;
		}
		const lowerName = cleanName.toLowerCase();
		const isOpenRouter = providerName === 'openRouter';

		// Common model context windows
		const contextWindows: Record<string, number> = {
			// OpenAI
			'gpt-4-turbo': 128000,
			'gpt-4-turbo-preview': 128000,
			'gpt-4-1106-preview': 128000,
			'gpt-4': 8192,
			'gpt-4-32k': 32768,
			'gpt-3.5-turbo': 16385,
			'gpt-3.5-turbo-16k': 16385,
			'o1-preview': 128000,
			'o1-mini': 128000,
			'o3-mini': 200000,
			// Anthropic
			'claude-3-opus': 200000,
			'claude-3-sonnet': 200000,
			'claude-3-haiku': 200000,
			'claude-3.5-sonnet': 200000,
			'claude-3.5-haiku': 200000,
			'claude-3.7-sonnet': 200000,
			// Google
			'gemini-pro': 32768,
			'gemini-1.5-pro': 1000000,
			'gemini-1.5-flash': 1000000,
			'gemini-2.0-flash': 1000000,
			'gemini-3-flash': 1000000,
			'gemini-3-flash-preview': 1000000,
			'gemini-3-pro-preview': 1000000,
			'gemini-3-pro': 1000000,
			'nemotron-3-nano': 1000000,
			// xAI Grok
			'grok-2': 131072,
			'grok-3': 131072,
			'grok-3-fast': 131072,
			'grok-3-mini': 131072,
			'grok-3-mini-fast': 131072,
			'grok-4': 256000,
			'grok-4-fast': 2000000,
			'grok-4.1-fast': 2000000,
			'grok-4.1-fast:free': 2000000,
			// DeepSeek
			'deepseek-v3': 128000,
			'deepseek-r1': 128000,
			// Ollama Cloud models
			'deepseek-v3.1:671b-cloud': 128000,
			'gpt-oss:20b-cloud': 128000,
			'gpt-oss:120b-cloud': 128000,
			'kimi-k2:1t-cloud': 128000,
			'kimi-k2-thinking:1t-cloud': 256000, // Kimi K2 Thinking has 256k context
			'kimi-k2-thinking:cloud': 256000, // Alias for kimi-k2-thinking:1t-cloud
			'qwen3-coder:480b-cloud': 128000,
			'minimax-m2:cloud': 128000,
			'minimax-m2.1:cloud': 200000, // Reduced from 204800 to be safe
			'glm-4.6': 128000,
			'glm-4.7': 203495,
			// Ollama models (common ones)
			'llama3.3': 128000,
			'llama3.1': 128000,
			'llama3.2': 128000,
			'llama3': 8192,
			'llama2': 4096,
			'mistral': 8192,
			'mixtral': 32768,
			'qwen': 32768,
			'qwen2': 32768,
			'codellama': 16384,
			'deepseek-coder': 16384,
			'phi': 2048,
			'gemma': 8192,
			'gemma2': 8192,
			// Other local models
			'yi': 4096,
			'solar': 4096,
		};

		// Try exact match first
		if (contextWindows[lowerName]) {
			return contextWindows[lowerName];
		}

		// Try to match by stripping everything after the first colon in lowerName
		const baseName = lowerName.split(':')[0];
		if (baseName && contextWindows[baseName]) {
			return contextWindows[baseName];
		}

		// Try partial match, prefer longer keys for better specificity
		const sortedKeys = Object.keys(contextWindows).sort((a, b) => b.length - a.length);
		for (const key of sortedKeys) {
			if (lowerName.includes(key)) {
				return contextWindows[key];
			}
		}

		// Default for OpenRouter models (most are 128k+)
		if (isOpenRouter) {
			voidDevWarn(`[TokenCountingService] Unknown OpenRouter model ${modelName}, defaulting to 128000`);
			return 128000;
		}

		// For Ollama and local models, default to 8k (more generous than 4k)
		// Most modern local models support at least 8k context
		const isLikelyLocal = lowerName.includes('ollama') ||
			lowerName.includes('local') ||
			lowerName.includes('llama') ||
			lowerName.includes('mistral');

		if (isLikelyLocal) {
			// Hard override for Minimax models on Ollama Cloud which advertise 1M but fail > 200k
			if (lowerName.includes('minimax')) {
				return 200000;
			}
			voidDevWarn(`[TokenCountingService] Unknown Ollama/local model ${modelName}, defaulting to 8192`);
			return 8192;
		}

		// llama.cpp servers run with whatever -c/--ctx-size the user launched them with, which
		// we can't infer from the model name. Default conservatively so rolling-window
		// compression engages and prevents the server from silently context-shift-truncating
		// long threads (a 131985 assumption would never trigger compression and let the server
		// drop tokens mid-conversation with no warning). Users running a larger context should
		// set it in Model Management (override), which is respected at the top of this function.
		if (providerName === 'llamaCpp') {
			voidDevWarn(`[TokenCountingService] Unknown llama.cpp model ${modelName}, defaulting context window to 8192. Set the real -c/--ctx-size in Model Management if your server uses a larger context.`);
			return 8192;
		}

		// Default to 131985 for unknown models (matches common modern model contexts)
		voidDevWarn(`[TokenCountingService] Unknown context window for ${modelName}, defaulting to 131985`);
		return 131985;
	}

	/**
	 * Calculate remaining tokens in context window (sync, uses estimation)
	 * Use for non-critical operations where speed is more important than accuracy.
	 */
	public getRemainingTokens(messages: LLMChatMessage[], modelName: string): number {
		const usedTokens = this.countMessagesTokens(messages, modelName);
		const contextWindow = this.getContextWindowSize(modelName);
		return Math.max(0, contextWindow - usedTokens);
	}

	/**
	 * Calculate remaining tokens in context window (async, uses accurate tiktoken)
	 * CRITICAL: Use this for compression decisions and context window checks.
	 */
	public async getRemainingTokensAsync(messages: LLMChatMessage[], modelName: string): Promise<number> {
		const usedTokens = await this.countMessagesTokensAsync(messages, modelName);
		const contextWindow = this.getContextWindowSize(modelName);
		return Math.max(0, contextWindow - usedTokens);
	}

	/**
	 * Check if messages fit within context window (sync, uses estimation)
	 */
	public fitsInContextWindow(messages: LLMChatMessage[], modelName: string): boolean {
		return this.getRemainingTokens(messages, modelName) > 0;
	}

	/**
	 * Check if messages fit within context window (async, uses accurate tiktoken)
	 * CRITICAL: Use this for critical operations before making LLM calls.
	 */
	public async fitsInContextWindowAsync(messages: LLMChatMessage[], modelName: string): Promise<boolean> {
		const remaining = await this.getRemainingTokensAsync(messages, modelName);
		return remaining > 0;
	}

	/**
	 * Check if compression is needed with accurate async token counting
	 * CRITICAL: Use this instead of sync version for compression decisions.
	 */
	public async needsCompressionAsync(messages: LLMChatMessage[], modelName: string, threshold: number = 0.8): Promise<boolean> {
		const currentTokens = await this.countMessagesTokensAsync(messages, modelName);
		const contextWindow = this.getContextWindowSize(modelName);
		const usage = currentTokens / contextWindow;
		return usage > threshold;
	}

	/**
	 * Get compression preview with accurate async token counting
	 * CRITICAL: Use this for accurate compression statistics.
	 */
	public async getCompressionPreviewAsync(messages: LLMChatMessage[], modelName: string, config?: Partial<CompressionConfig>): Promise<CompressionStats> {
		// This will be called by ContextCompressionService
		// The implementation is in ContextCompressionService but uses this for token counting
		const currentTokens = await this.countMessagesTokensAsync(messages, modelName);
		const contextWindow = this.getContextWindowSize(modelName);
		const effectiveContextWindow = contextWindow - this.getLargeContextBuffer(contextWindow);
		const targetTokens = Math.floor((effectiveContextWindow - 8192) * 0.85);

		return {
			originalTokens: currentTokens,
			originalMessageCount: messages.length,
			targetTokens,
			finalTokens: 0,
			finalMessageCount: 0,
			messagesRemoved: 0,
			messagesSummarized: 0,
			toolResultsTruncated: 0,
			compressionRatio: Math.round((currentTokens / targetTokens) * 100)
		};
	}

	/**
	 * Estimate tokens for a completion response
	 * This is a rough estimate - actual tokens will vary
	 */
	public estimateCompletionTokens(promptTokens: number, modelName: string): number {
		const contextWindow = this.getContextWindowSize(modelName);
		// Reserve 25% of remaining window for completion, or max 4096 tokens
		const remaining = contextWindow - promptTokens;
		return Math.min(4096, Math.floor(remaining * 0.25));
	}

	/**
	 * Get conservative buffer for very large context windows (1M+ tokens)
	 * Large models like Gemini 1.5 have non-linear tokenization for very large inputs
	 */
	public getLargeContextBuffer(contextWindow: number): number {
		// For models with 1M+ context, use a more conservative buffer
		// to account for non-linear tokenization and encoding overhead
		if (contextWindow >= 1000000) {
			// 5% buffer for 1M+ context windows
			return Math.floor(contextWindow * 0.05);
		} else if (contextWindow >= 500000) {
			// 3% buffer for 500k-1M context windows
			return Math.floor(contextWindow * 0.03);
		} else if (contextWindow >= 200000) {
			// 2% buffer for 200k-500k context windows
			return Math.floor(contextWindow * 0.02);
		} else {
			// 1% buffer for smaller context windows
			return Math.floor(contextWindow * 0.01);
		}
	}

	/**
	 * Estimate tokens with improved accuracy for very large contexts
	 * Accounts for non-linear tokenization in large context models
	 */
	public estimateTokensForLargeContext(text: string, modelName: string): number {
		const contextWindow = this.getContextWindowSize(modelName);
		
		// For very large contexts, tokenization becomes less efficient
		// This is an empirical observation for models with 1M+ context
		if (contextWindow >= 1000000) {
			// Large context models often have 3-3.5 chars/token for typical content
			// Add overhead for special formatting and non-linear encoding
			return Math.ceil(text.length / 3.2) + 50;
		} else if (contextWindow >= 500000) {
			return Math.ceil(text.length / 3.5) + 30;
		} else {
			// Standard estimate for smaller contexts
			return Math.ceil(text.length / 4) + 10;
		}
	}

	/**
	 * Get effective context window accounting for large context buffers
	 */
	public getEffectiveContextWindow(modelName: string): number {
		const contextWindow = this.getContextWindowSize(modelName);
		const buffer = this.getLargeContextBuffer(contextWindow);
		return contextWindow - buffer;
	}

	/**
	 * Get dynamically cached context window for a model
	 */
	private _getDynamicContextWindow(modelName: string): number | undefined {
		const cached = this._dynamicContextWindows.get(modelName);
		if (!cached) return undefined;

		// Check if cache is still valid
		if (Date.now() - cached.timestamp > this.CONTEXT_WINDOW_CACHE_TTL_MS) {
			this._dynamicContextWindows.delete(modelName);
			return undefined;
		}

		return cached.window;
	}

	/**
	 * Set dynamic context window cache
	 */
	public setDynamicContextWindow(modelName: string, contextWindow: number): void {
		this._dynamicContextWindows.set(modelName, {
			window: contextWindow,
			timestamp: Date.now()
		});
		voidDevLog(`[TokenCountingService] Cached context window for ${modelName}: ${contextWindow}`);
	}

	/**
	 * Fetch context window from model API dynamically.
	 * This should be called when static lookup fails.
	 * Currently returns undefined but provides framework for future API integration.
	 */
	public async fetchDynamicContextWindow(modelName: string): Promise<number | undefined> {
		// Check cache first
		const cached = this._getDynamicContextWindow(modelName);
		if (cached !== undefined) {
			return cached;
		}

		try {
			// NOTE: Dynamic context window fetching via provider APIs is a planned enhancement
			// - OpenAI: GET /v1/models/{model}
			// - Anthropic: GET /v1/models
			// - OpenRouter: GET /api/v1/models
			// - Ollama: GET /api/tags or /api/show

			// For now, we rely on static mappings
			// This framework allows easy addition of dynamic fetching later

			// Attempt to infer from model name patterns
			const inferredWindow = this._inferContextWindowFromName(modelName);
			if (inferredWindow) {
				this.setDynamicContextWindow(modelName, inferredWindow);
				return inferredWindow;
			}

			return undefined;
		} catch (error) {
			voidDevWarn(`[TokenCountingService] Failed to fetch dynamic context window for ${modelName}:`, error);
			return undefined;
		}
	}

	/**
	 * Infer context window from model name patterns
	 * Uses common naming conventions to make educated guesses
	 */
	private _inferContextWindowFromName(modelName: string): number | undefined {
		const lowerName = modelName.toLowerCase();

		// Check for context window indicators in model name
		if (lowerName.includes('128k') || lowerName.includes('128000')) return 128000;
		if (lowerName.includes('200k') || lowerName.includes('200000')) return 200000;
		if (lowerName.includes('256k') || lowerName.includes('256000')) return 256000;
		if (lowerName.includes('512k') || lowerName.includes('512000')) return 512000;
		if (lowerName.includes('1m') || lowerName.includes('1000000')) return 1000000;
		if (lowerName.includes('2m') || lowerName.includes('2000000')) return 2000000;

		// Check for model family patterns
		if (lowerName.includes('claude-3')) return 200000; // All Claude 3 models have 200k
		if (lowerName.includes('claude-3.5')) return 200000;
		if (lowerName.includes('claude-3.7')) return 200000;
		if (lowerName.includes('gemini-1.5')) return 1000000; // Gemini 1.5 has 1M
		if (lowerName.includes('gemini-2')) return 1000000;
		if (lowerName.includes('gemini-3')) return 1000000;

		return undefined;
	}

	/**
	 * Clear expired cache entries
	 */
	public clearExpiredCaches(): void {
		const now = Date.now();
		for (const [key, data] of this._dynamicContextWindows.entries()) {
			if (now - data.timestamp > this.CONTEXT_WINDOW_CACHE_TTL_MS) {
				this._dynamicContextWindows.delete(key);
			}
		}
	}

	/**
	 * Dispose and clean up resources
	 */
	override dispose(): void {
		this._countCache.clear();
		this._historyCache.clear();
		this._cacheTimestamps.clear();
		this._dynamicContextWindows.clear();
		this._pendingAsyncCounts.clear();
		super.dispose();
	}
}

registerSingleton(ITokenCountingService, TokenCountingService, InstantiationType.Eager);
