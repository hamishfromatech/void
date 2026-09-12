/*--------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { Disposable, IDisposable } from '../../../../base/common/lifecycle.js';
import { registerSingleton, InstantiationType } from '../../../../platform/instantiation/common/extensions.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { IModelService } from '../../../../editor/common/services/model.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';

import { URI } from '../../../../base/common/uri.js';
import { Emitter, Event } from '../../../../base/common/event.js';
import { ILLMMessageService } from '../common/sendLLMMessageService.js';
import { voidDevLog, voidDevWarn } from '../common/devLog.js';
import { chat_userMessageContent, isABuiltinToolName } from '../common/prompt/prompts.js';
import { AnthropicReasoning, getErrorMessage, normalizeStopReason, RawToolCallObj, RawToolParamsObj } from '../common/sendLLMMessageTypes.js';
import { generateUuid } from '../../../../base/common/uuid.js';
import { FeatureName, ModelSelection, ModelSelectionOptions } from '../common/voidSettingsTypes.js';
import { IVoidSettingsService } from '../common/voidSettingsService.js';
import { approvalTypeOfBuiltinToolName, BuiltinToolCallParams, BuiltinToolName, ToolCallParams, ToolName, ToolResult } from '../common/toolsServiceTypes.js';
import { shouldAutoApproveTerminalTool, TerminalAutoApproveSettings } from '../common/terminalApproval.js';
import { IToolsService } from './toolsService.js';
import { CancellationToken, CancellationTokenSource } from '../../../../base/common/cancellation.js';
import { ILanguageFeaturesService } from '../../../../editor/common/services/languageFeatures.js';
import { ChatMessage, CheckpointEntry, CodespanLocationLink, StagingSelectionItem, ToolMessage, ImageAttachment, StudentSession, StudentExercise, ActiveWorkflow, QueueBehavior, CompactionSnapshot } from '../common/chatThreadServiceTypes.js';
import { Position } from '../../../../editor/common/core/position.js';
import { IMetricsService } from '../common/metricsService.js';
import { shorten } from '../../../../base/common/labels.js';
import { IVoidModelService } from '../common/voidModelService.js';
import { findLast, findLastIdx } from '../../../../base/common/arraysFind.js';
import { IEditCodeService } from './editCodeServiceInterface.js';
import { VoidFileSnapshot, DiffBasedCheckpoint, createDiffBasedCheckpoint, applyDiffBasedCheckpoint } from '../common/editCodeServiceTypes.js';
import { INotificationService, Severity } from '../../../../platform/notification/common/notification.js';
import { truncate } from '../../../../base/common/strings.js';
import { THREAD_STORAGE_KEY, THREAD_STORAGE_KEY_PREFIX, THREAD_STORAGE_VERSION_KEY, CURRENT_THREAD_STORAGE_VERSION } from '../common/storageKeys.js';
import { truncateToolOutputWithNotice } from '../common/truncateToolOutput.js';
import { AgentLoopEvent } from './agentLoopEvents.js';
import { IVisionService } from './visionService.js';
import { IConvertToLLMMessageService, UsageAnchor } from './convertToLLMMessageService.js';
import { IToolOrchestrationService, OrchestrationResult } from './toolOrchestrationService.js';
import { timeout } from '../../../../base/common/async.js';
import { deepClone } from '../../../../base/common/objects.js';
import { IWorkspaceContextService } from '../../../../platform/workspace/common/workspace.js';
import { IDirectoryStrService } from '../common/directoryStrService.js';
import { IFileService } from '../../../../platform/files/common/files.js';
import { IMCPService } from '../common/mcpService.js';
import { IACPService } from '../common/acpService.js';
import { IComposioService } from '../common/composioService.js';
import { RawMCPToolCall } from '../common/mcpServiceTypes.js';
import { ACPRunAgentResponse } from '../common/acpServiceTypes.js';
import { StreamingXMLParser, ReActPhase } from './streamingXMLParser.js';
import { ToonService } from '../common/toonService.js';
import { IHookService } from '../common/hookService.js';
import { ISubagentService } from './subagentService.js';
import { triggerCompressionNotification } from './react/src/util/compressionState.js';


// related to retrying when LLM message has error
const CHAT_RETRIES = 3 // Number of retries for LLM errors (including empty responses)
const RETRY_DELAY_BASE = 1000 // Base delay between retries in milliseconds (exponential backoff: 1s, 2s, 4s)
const MAX_STOP_HOOK_POKES = 8 // Cap on Stop-hook-forced continuation turns to avoid runaway /goal loops
export const AUTO_CONTINUE_CHAR_THRESHOLD = 200; // Still used by UI auto-continue

// MEMORY OPTIMIZATION: Maximum messages per thread to prevent unbounded memory growth
// Each message can be several KB with tool results, images, etc.
// Increased from 15 to 100 with UI virtualization for performance
const MAX_MESSAGES_PER_THREAD = 100;

// MEMORY OPTIMIZATION: Maximum checkpoints per thread to prevent memory bloat
const MAX_CHECKPOINTS_PER_THREAD = 5;

// MEMORY OPTIMIZATION: Maximum tool call history per thread to prevent unbounded memory growth
// Tool calls can store large params and results
const MAX_TOOL_CALL_HISTORY_PER_THREAD = 100;

// MEMORY OPTIMIZATION: Maximum length of tool result strings to prevent memory bloat
const MAX_TOOL_RESULT_LENGTH = 10000;

// MEMORY OPTIMIZATION: Maximum images per message to prevent memory bloat
// Base64 images can be several MB each
const MAX_IMAGES_PER_MESSAGE = 5;

// MEMORY OPTIMIZATION: Maximum message queue size per thread
const MAX_MESSAGE_QUEUE_PER_THREAD = 10;

// Detection of provider context-length errors, used to decide whether an LLM
// error should trigger token-ratio adjustment + compression retry. Deliberately
// narrow: bare substrings like "400" or "token" also match auth failures
// ("invalid token"), timeouts ("context deadline exceeded"), and unrelated 4xx
// responses, which previously got misclassified as context-length errors.
const CONTEXT_LENGTH_ERROR_RE = /maximum context length|context length|context window|prompt is too long|input token count[^.]*exceeds|too many (input )?tokens|exceeds[^.]*tokens? (limit|maximum)|tokens? (limit|maximum)[^.]*exceed|reduce the length|too long/i;

// MEMORY OPTIMIZATION: Maximum total size of all images in a message (10MB)
const MAX_TOTAL_IMAGE_SIZE_MB = 10;

// SIZE-BASED MEMORY LIMITS: New limits to prevent memory bloat from large content
const MAX_MESSAGE_SIZE_KB = 500; // Max 500KB per message (includes tool results)
const MAX_THREAD_SIZE_MB = 50;   // Max 50MB total per thread

// DYNAMIC PARALLEL TOOL CALLING: Base sets for determining parallel safety
// These are the default read-only tools that CAN be parallel-safe, but actual
// safety is determined dynamically based on parameters and execution context.
const READ_ONLY_TOOLS: ReadonlySet<string> = new Set([
	'read_file',
	'outline_file',
	'ls_dir',
	'get_dir_tree',
	'search_pathnames_only',
	'search_for_files',
	'search_in_file',
	'read_lint_errors',
	'fast_context',
	'codebase_search',
])

// Planning/task tools are safe to run in parallel — they operate on in-memory
// state with no file-system, terminal, or git conflicts.
const PARALLEL_SAFE_WRITE_TOOLS: ReadonlySet<string> = new Set([
	'create_todo',
	'update_todo',
	'add_todos',
])

// Tools that are NEVER safe to run in parallel due to global state
const SEQUENTIAL_ONLY_TOOLS: ReadonlySet<string> = new Set([
	'run_command',      // Terminal state is shared
	'run_code',         // Process execution state is shared
	'edit_file',        // File editing can conflict
	'edit_files',       // Multi-file editing can conflict
	'rewrite_file',     // File rewriting can conflict
	'delete',           // Deletions can conflict
	'git_commit',       // Git state is shared
	'git_diff',         // Git state is shared
	'apply_diff',       // File editing can conflict
])

/**
 * Determine if multiple tool calls can safely execute in parallel.
 *
 * Dynamic safety analysis based on:
 * 1. Tool type (read-only vs write)
 * 2. Target resources (file paths, directories)
 * 3. Approval status (auto-approved vs requires approval)
 * 4. MCP vs built-in tools
 *
 * @param toolCalls - Array of tool calls to analyze
 * @returns Object with parallelSafe and sequential arrays
 */
export function analyzeParallelToolSafety(
	toolCalls: RawToolCallObj[]
): { parallelSafe: RawToolCallObj[]; sequential: RawToolCallObj[] } {
	const parallelSafe: RawToolCallObj[] = []
	const sequential: RawToolCallObj[] = []

	// Single tool is always safe (no concurrency concerns)
	if (toolCalls.length <= 1) {
		return { parallelSafe: toolCalls, sequential: [] }
	}

	// Group tool calls by their target resource (file path, directory, etc.)
	const toolGroupsByResource = new Map<string, RawToolCallObj[]>()

	for (const toolCall of toolCalls) {
		const resourceKey = getToolResourceKey(toolCall)

		if (!toolGroupsByResource.has(resourceKey)) {
			toolGroupsByResource.set(resourceKey, [])
		}
		toolGroupsByResource.get(resourceKey)!.push(toolCall)
	}

	// Analyze each group
	for (const [, group] of toolGroupsByResource.entries()) {
		for (const toolCall of group) {
			const canBeParallel = canToolRunParallel(toolCall)

			if (canBeParallel) {
				parallelSafe.push(toolCall)
			} else {
				sequential.push(toolCall)
			}
		}
	}

	// Additional safety check: if multiple tools target the same file,
	// only read-only tools can run in parallel
	for (const [resourceKey, group] of toolGroupsByResource.entries()) {
		if (group.length > 1 && resourceKey.startsWith('file:')) {
			// Multiple tools targeting same file - check for write conflicts
			const hasWriteTool = group.some(tc => isWriteTool(tc.name))
			if (hasWriteTool) {
				// Move all tools in this group to sequential
				const writeTools = group.filter(tc => isWriteTool(tc.name))
				const readTools = group.filter(tc => !isWriteTool(tc.name))

				// Remove from their current arrays
				for (const tc of writeTools) {
					const idx = parallelSafe.indexOf(tc)
					if (idx !== -1) parallelSafe.splice(idx, 1)
				}
				for (const tc of readTools) {
					const idx = parallelSafe.indexOf(tc)
					if (idx !== -1) parallelSafe.splice(idx, 1)
				}

				// Add all to sequential
				sequential.push(...group)
			}
		}
	}

	return { parallelSafe, sequential }
}

/**
 * Get a resource key for a tool call (used for conflict detection)
 */
function getToolResourceKey(toolCall: RawToolCallObj): string {
	const params = toolCall.rawParams || {}

	// File-based tools
	if (params.filePath) return `file:${params.filePath}`
	if (params.path) return `file:${params.path}`
	if (params.file) return `file:${params.file}`
	if (params.directory) return `dir:${params.directory}`
	if (params.dir) return `dir:${params.dir}`

	// Command-based tools
	if (params.command) return `command:${params.command.substring(0, 50)}`

	// Search-based tools
	if (params.query) return `search:${params.query.substring(0, 50)}`
	if (params.pattern) return `pattern:${params.pattern}`

	// Default: tool name as key
	return `tool:${toolCall.name}`
}

/**
 * Check if a tool is a write operation (modifies state)
 */
function isWriteTool(toolName: string): boolean {
	return SEQUENTIAL_ONLY_TOOLS.has(toolName) ||
		toolName === 'create_file_or_folder' ||
		toolName === 'edit_file' ||
		toolName === 'edit_files' ||
		toolName === 'rewrite_file' ||
		toolName === 'delete' ||
		toolName === 'apply_diff'
}

/**
 * Determine if a single tool can potentially run in parallel
 * (actual parallel safety also depends on other concurrent tools)
 */
function canToolRunParallel(toolCall: RawToolCallObj): boolean {
	// Sequential-only tools are never safe in parallel
	if (SEQUENTIAL_ONLY_TOOLS.has(toolCall.name)) {
		return false
	}

	// Read-only tools are generally safe for parallel execution
	if (READ_ONLY_TOOLS.has(toolCall.name)) {
		return true
	}

	// Planning/task tools are safe in parallel (in-memory state, no resource conflicts)
	if (PARALLEL_SAFE_WRITE_TOOLS.has(toolCall.name)) {
		return true
	}

	// create_file_or_folder is safe in parallel ONLY if targeting different paths
	if (toolCall.name === 'create_file_or_folder') {
		return true // Path conflict detection handled by analyzeParallelToolSafety
	}

	// Unknown tools default to sequential for safety
	return false
}

const splitThinkTags = (input: string): { displayText: string; reasoningText: string } => {
	if (!input) {
		return { displayText: '', reasoningText: '' }
	}
	// Treat the special placeholder as empty content
	if (input === '(empty message)') {
		return { displayText: '', reasoningText: '' }
	}

	const reasoningParts: string[] = []

	// Helper to extract content from both closed and unclosed tags
	const extractFromTags = (text: string, openTag: string, closeTag: string): string => {
		let currentText = text
		let lastIndex = 0

		// Optimization: check if tag exists at all before doing complex work
		if (!text.includes(openTag)) return text;

		while (true) {
			const openIdx = currentText.indexOf(openTag, lastIndex)
			if (openIdx === -1) break

			const closeIdx = currentText.indexOf(closeTag, openIdx + openTag.length)

			if (closeIdx !== -1) {
				// Found closed tag
				const content = currentText.substring(openIdx + openTag.length, closeIdx).trim()
				if (content) reasoningParts.push(content)
				// KEEP the text after the close tag
				currentText = currentText.substring(0, openIdx) + currentText.substring(closeIdx + closeTag.length)
				lastIndex = openIdx
			} else {
				// Found unclosed tag - take everything until the end
				const content = currentText.substring(openIdx + openTag.length).trim()
				if (content) reasoningParts.push(content)

				// but only return the text BEFORE the tag as display content
				currentText = currentText.substring(0, openIdx)
				break // No more closed tags possible after an unclosed one
			}
		}
		return currentText
	}

	let remainingText = input
	remainingText = extractFromTags(remainingText, '<think>', '</think>')
	remainingText = extractFromTags(remainingText, '<reasoning>', '</reasoning>')

	return { 
		displayText: remainingText.trim(), 
		reasoningText: reasoningParts.join('\n\n') 
	}
}

const mergeReasoningContent = (existing?: string | null, fromTags?: string | null): string => {
	const primary = (existing ?? '').trim()
	const secondary = (fromTags ?? '').trim()
	if (!primary) return secondary
	if (!secondary) return primary

	// PERFORMANCE OPTIMIZATION: Avoid expensive includes on large strings.
	// If the strings are similar in length, they might be the same.
	if (primary === secondary) return primary

	// If one is significantly longer, it likely contains the other.
	// We only check for inclusion if the length difference is small or if strings are small.
	if (primary.length < 1000 && secondary.length < 1000) {
		if (primary.includes(secondary)) return primary
		if (secondary.includes(primary)) return secondary
	}

	// Otherwise, we assume they are different and append them.
	// To prevent unbounded growth, we don't re-append the same secondary many times.
	// This is a heuristic: if primary ends with the start of secondary, we might want to merge,
	// but simple appending is safer and faster for most cases.
	return `${primary}\n\n${secondary}`.trim()
}

// Task planning system inspired by Cursor's approach
export interface TaskPlan {
	id: string
	description: string
	status: 'pending' | 'in_progress' | 'completed' | 'blocked'
	dependencies?: string[]
	created_at: number
	completed_at?: number
}

const partitionReasoningContent = (fullText: string, existingReasoning?: string | null): { displayText: string, reasoningText: string } => {
	const { displayText, reasoningText } = splitThinkTags(fullText)
	const mergedReasoning = mergeReasoningContent(existingReasoning, reasoningText)
	const normalizedDisplay = displayText.replace(/[\s\u00a0]+$/, '')
	return {
		displayText: normalizedDisplay,
		reasoningText: mergedReasoning,
	}
}

/**
 * Heuristic to detect if the LLM message sounds "unfinished" or like it intended to call a tool but didn't.
 * Returns 'silent' if we should auto-continue without a poke, 'poke' if we should add a user message.
 */
const detectDanglingAgenticIntent = (text: string, reasoning: string): 'none' | 'silent' | 'poke' => {
	const combined = (text + ' ' + reasoning).trim();
	if (!combined) return 'none';

	// Patterns where the model is CLEARLY just about to emit a tool call or XML block
	// We can silently continue these to avoid UI noise
	const silentPatterns = [
		// Ends with Action: but no content (ReAct style)
		/Action:\s*$/i,
		// Ends with a tool call opening but no content
		/<function_calls>\s*$/is,
		/<invoke\s+name="[^"]*"\s*>\s*$/is,
		// Ends with "I will now use the [X] tool to [Y]:"
		/(?:I will|I'll|Let me|I'm going to|I'll now|I will now)\s+(?:use|call|invoke|execute)\s+(?:the\s+)?(?:[a-zA-Z0-9_-]+)\s+tool\s+to\s+[^.!?]*:\s*$/i,
		// Ends with a very specific "About to act" pattern
		/(?:Based on the above,|Therefore,|So,|I'll start by|I will begin by)\s+(?:I will|I'll|I'm going to)\s+(?:now\s+)?(?:read|edit|search|run|check|fix|update|create|delete)\s+[^.!?]*:\s*$/i,
	];

	if (silentPatterns.some(p => p.test(combined))) return 'silent';

	// Patterns indicating the LLM intended to call a tool but stopped at a more ambiguous point
	const pokePatterns = [
		// Interrupted intent to use a common tool at the end of a sentence
		/(?:I will|I'll|Let me|I'm going to|I'll now|I will now)\s+(?:read|edit|search|run|check|fix|update|create|delete|list|get|inspect|use|call|invoke|open|execute)\b[^.!?]*$/i,
		// ReAct style thought without action
		/Thought:\s*(?!.*Action:)/is,
		// Unclosed XML tags (already started but not finished)
		/<function_calls>(?!.*<\/function_calls>)/is,
		/<invoke\s+name="[^"]*"(?!.*<\/invoke>)/is,
		// Ends with a plan step but no action follows
		/Plan:\s*\d+\.\s+.*$/is,
	];

	if (pokePatterns.some(p => p.test(combined))) return 'poke';

	return 'none';
};


const findStagingSelectionIndex = (currentSelections: StagingSelectionItem[] | undefined, newSelection: StagingSelectionItem): number | null => {
	if (!currentSelections) return null

	for (let i = 0; i < currentSelections.length; i += 1) {
		const s = currentSelections[i]

		if (s.uri.fsPath !== newSelection.uri.fsPath) continue

		if (s.type === 'File' && newSelection.type === 'File') {
			return i
		}
		if (s.type === 'CodeSelection' && newSelection.type === 'CodeSelection') {
			if (s.uri.fsPath !== newSelection.uri.fsPath) continue
			// if there's any collision return true
			const [oldStart, oldEnd] = s.range
			const [newStart, newEnd] = newSelection.range
			if (oldStart !== newStart || oldEnd !== newEnd) continue
			return i
		}
		if (s.type === 'Folder' && newSelection.type === 'Folder') {
			return i
		}
	}
	return null
}


/*

Store a checkpoint of all "before" files on each x.
x's show up before user messages and LLM edit tool calls.

x     A          (edited A -> A')
(... user modified changes ...)
User message

x     A' B C     (edited A'->A'', B->B', C->C')
LLM Edit
x
LLM Edit
x
LLM Edit


INVARIANT:
A checkpoint appears before every LLM message, and before every user message (before user really means directly after LLM is done).
*/


type UserMessageType = ChatMessage & { role: 'user' }
type UserMessageState = UserMessageType['state']
const defaultMessageState: UserMessageState = {
	stagingSelections: [],
	isBeingEdited: false,
}

// a 'thread' means a chat message history

type WhenMounted = {
	textAreaRef: { current: HTMLTextAreaElement | null }; // the textarea that this thread has, gets set in SidebarChat
	scrollToBottom: () => void;
}



export type ThreadType = {
	id: string; // store the id here too
	createdAt: string; // ISO string
	lastModified: string; // ISO string
	name?: string; // Optional user-defined thread name

	messages: ChatMessage[];
	filesWithUserChanges: Set<string>;

	// this doesn't need to go in a state object, but feels right
	state: {
		currCheckpointIdx: number | null; // the latest checkpoint we're at (null if not at a particular checkpoint, like if the chat is streaming, or chat just finished and we haven't clicked on a checkpt)

		stagingSelections: StagingSelectionItem[];
		focusedMessageIdx: number | undefined; // index of the user message that is being edited (undefined if none)

		linksOfMessageIdx: { // eg. link = linksOfMessageIdx[4]['RangeFunction']
			[messageIdx: number]: {
				[codespanName: string]: CodespanLocationLink
			}
		}


		mountedInfo?: {
			whenMounted: Promise<WhenMounted>
			_whenMountedResolver: (res: WhenMounted) => void
			mountedIsResolvedRef: { current: boolean };
		}


		autoContinueEnabled: boolean;

		// Workflow tracking
		activeWorkflow: ActiveWorkflow | null;
		queueBehavior: QueueBehavior;

		// Student mode session state
		studentSession?: StudentSession;

		// Voice mode state
		voiceModeActive: boolean;

		// Skills system: map of skill name to its instructions/content
		loadedSkills: { [skillName: string]: string };

		// `/compact` snapshot: when present, the send path replaces the leading
		// `compactedChatMessageCount` messages with `summaryText`. See CompactionSnapshot.
		compaction?: CompactionSnapshot;
	};
}

type ChatThreads = {
	[id: string]: undefined | ThreadType;
}


export type ThreadsState = {
	allThreads: ChatThreads;
	currentThreadId: string; // intended for internal use only
}

export type IsRunningType =
	| 'LLM' // the LLM is currently streaming
	| 'tool' // whether a tool is currently running
	| 'awaiting_user' // awaiting user call
	| 'idle' // nothing is running now, but the chat should still appear like it's going (used in-between calls)
	| undefined

export type ThreadStreamState = {
	[threadId: string]: undefined | {
		isRunning: undefined;
		error?: { message: string, fullError: Error | null, };
		llmInfo?: undefined;
		toolInfo?: undefined;
		interrupt?: undefined;
		tokenUsage?: { used: number, total: number, percentage: number };
	} | { // an assistant message is being written
		isRunning: 'LLM';
		error?: undefined;
		llmInfo: {
			displayContentSoFar: string;
			reasoningSoFar: string;
			toolCallsSoFar: RawToolCallObj[] | null;
			_rawTextBeforeStripping?: string; // For XML tool call detection in UI
			reactPhase?: ReActPhase | null; // Current ReAct phase for UI
			textDelta?: string; // Raw chunk for direct streaming
			reasoningDelta?: string; // Raw reasoning chunk for direct streaming
		};
		toolInfo?: undefined;
		interrupt: Promise<() => void>; // calling this should have no effect on state - would be too confusing. it just cancels the tool
		tokenUsage?: { used: number, total: number, percentage: number };
	} | { // a tool is being run
		isRunning: 'tool';
		error?: undefined;
		llmInfo?: undefined;
		toolInfo: {
			toolName: ToolName;
			toolParams: ToolCallParams<ToolName>;
			id: string;
			content: string;
			rawParams: RawToolParamsObj;
			mcpServerName: string | undefined;
		};
		interrupt: Promise<() => void>;
		tokenUsage?: { used: number, total: number, percentage: number };
	} | {
		isRunning: 'awaiting_user';
		error?: undefined;
		llmInfo?: undefined;
		toolInfo?: undefined;
		interrupt?: undefined;
		tokenUsage?: { used: number, total: number, percentage: number };
	} | {
		isRunning: 'idle';
		error?: undefined;
		llmInfo?: undefined;
		toolInfo?: undefined;
		interrupt: 'not_needed' | Promise<() => void>; // calling this should have no effect on state - would be too confusing. it just cancels the tool
		tokenUsage?: { used: number, total: number, percentage: number };
		stopReason?: string; // The LLM stop_reason/finish_reason from the last response
		cachedTokens?: number; // Prompt tokens served from the server's cache (e.g. llama.cpp cache_n / OpenAI cached_tokens), surfaced post-completion
	}
}

const newThreadObject = () => {
	const now = new Date().toISOString()
	return {
		id: generateUuid(),
		createdAt: now,
		lastModified: now,
		messages: [],
		state: {
			currCheckpointIdx: null,
			stagingSelections: [],
			focusedMessageIdx: undefined,
			linksOfMessageIdx: {},
			autoContinueEnabled: false,
			activeWorkflow: null,
			queueBehavior: 'wait_for_workflow',
			voiceModeActive: false,
			loadedSkills: {},
		},
		filesWithUserChanges: new Set()
	} satisfies ThreadType
}






export interface IChatThreadService {
	readonly _serviceBrand: undefined;

	readonly state: ThreadsState;
	readonly streamState: ThreadStreamState; // not persistent

	onDidChangeCurrentThread: Event<void>;
	onDidChangeStreamState: Event<{ threadId: string }>;
	onDidChangeMessageQueue: Event<{ threadId: string }>;

	getCurrentThread(): ThreadType;
	openNewThread(): void;
	switchToThread(threadId: string): void;

	// thread selector
	deleteThread(threadId: string): void;
	duplicateThread(threadId: string): void;
	setThreadName(threadId: string, name: string): void;

	// exposed getters/setters
	// these all apply to current thread
	getCurrentMessageState: (messageIdx: number) => UserMessageState
	setCurrentMessageState: (messageIdx: number, newState: Partial<UserMessageState>) => void
	getCurrentThreadState: () => ThreadType['state']
	setCurrentThreadState: (newState: Partial<ThreadType['state']>) => void

	// you can edit multiple messages - the one you're currently editing is "focused", and we add items to that one when you press cmd+L.
	getCurrentFocusedMessageIdx(): number | undefined;
	isCurrentlyFocusingMessage(): boolean;
	setCurrentlyFocusedMessageIdx(messageIdx: number | undefined): void;

	popStagingSelections(numPops?: number): void;
	addNewStagingSelection(newSelection: StagingSelectionItem): void;

	dangerousSetState: (newState: ThreadsState) => void;
	resetState: () => void;

	// // current thread's staging selections
	// closeCurrentStagingSelectionsInMessage(opts: { messageIdx: number }): void;
	// closeCurrentStagingSelectionsInThread(): void;

	// codespan links (link to symbols in the markdown)
	getCodespanLink(opts: { codespanStr: string, messageIdx: number, threadId: string }): CodespanLocationLink | undefined;
	addCodespanLink(opts: { newLinkText: string, newLinkLocation: CodespanLocationLink, messageIdx: number, threadId: string }): void;
	generateCodespanLink(opts: { codespanStr: string, threadId: string }): Promise<CodespanLocationLink>;
	getRelativeStr(uri: URI): string | undefined

	// entry pts
	abortRunning(threadId: string): Promise<void>;
	dismissStreamError(threadId: string): void;

	// call to edit a message
	editUserMessageAndStreamResponse({ userMessage, messageIdx, threadId }: { userMessage: string, messageIdx: number, threadId: string }): Promise<void>;

	// call to add a message
	addUserMessageAndStreamResponse({ userMessage, threadId, images, selections }: { userMessage: string, threadId: string, images?: ImageAttachment[], selections?: StagingSelectionItem[] }): Promise<void>;

	// call to add a task to a thread's task plan
	createTask(threadId: string, description: string, dependencies?: string[]): string;
	getTaskPlan(threadId: string): TaskPlan[];
	updateTaskStatus(threadId: string, taskId: string, status: TaskPlan['status']): void;
	deleteTask(threadId: string, taskId: string): void;
	clearTaskPlan(threadId: string): void;

	// approve/reject/skip
	approveLatestToolRequest(threadId: string, toolId?: string): void;
	rejectLatestToolRequest(threadId: string, toolId?: string): void;
	skipLatestToolRequest(threadId: string, toolId?: string): void;
	submitToolResult(threadId: string, toolId: string, result: any): void;

	// jump to history
	jumpToCheckpointBeforeMessageIdx(opts: { threadId: string, messageIdx: number, jumpToUserModified: boolean }): void;

	// Message operations
	deleteMessagesFromIndex(threadId: string, messageIdx: number): void;
	retryFromMessage(threadId: string, messageIdx: number): Promise<void>;
	copyMessageContent(threadId: string, messageIdx: number): string;

	// /compact — manually compress the thread's context into an LLM-generated summary.
	// `focusInstructions` optionally guides what the summary preserves (e.g. "the auth
	// bug fix"). Persists a CompactionSnapshot on the thread; the original messages are
	// retained for rewind. `clear` removes the snapshot so the full history is sent again.
	compactThread(threadId: string, focusInstructions?: string): Promise<void>;
	clearCompaction(threadId: string): void;

	// Message queue
	getQueuedMessagesCount(threadId: string): number;
	getQueuedMessages(threadId: string): Array<{ userMessage: string, selections?: StagingSelectionItem[], images?: ImageAttachment[] }>;
	removeQueuedMessage(threadId: string, index: number): void;
	clearMessageQueue(threadId: string): void;
	forceSendQueuedMessage(threadId: string, index: number): Promise<void>;

	// Auto-continue preference
	getAutoContinuePreference(threadId: string): boolean;
	setAutoContinuePreference(threadId: string, enabled: boolean): void;

	focusCurrentChat: (timeout?: number) => Promise<void>;
	blurCurrentChat: () => Promise<void>;

	// Student mode session
	getStudentSession(threadId: string): StudentSession | undefined;
	initStudentSession(threadId: string): StudentSession;
	addExercise(threadId: string, exercise: Omit<StudentExercise, 'hintLevel' | 'status' | 'createdAt'>): StudentExercise;
	updateExerciseHintLevel(threadId: string, exerciseId: string): number;
	completeExercise(threadId: string, exerciseId: string): void;
	addConceptLearned(threadId: string, concept: string): void;

	// Skills
	loadSkill(threadId: string, skillName: string, instructions: string): void;

	// Workflow management
	getActiveWorkflow(threadId: string): ThreadType['state']['activeWorkflow'];
	setActiveWorkflowStatus(threadId: string, status: ActiveWorkflow['status']): void;
	clearWorkflow(threadId: string): void;

	// Composio trigger handling
	handleComposioTrigger(event: {
		triggerSlug: string;
		userId: string;
		payload: Record<string, unknown>;
		metadata: { webhookId: string; triggerId: string; timestamp: string };
	}): void;
}

export const IChatThreadService = createDecorator<IChatThreadService>('chatThreadService');
class ChatThreadService extends Disposable implements IChatThreadService {
	_serviceBrand: undefined;

	// this fires when the current thread changes at all (a switch of currentThread, or a message added to it, etc)
	private readonly _onDidChangeCurrentThread = new Emitter<void>();
	readonly onDidChangeCurrentThread: Event<void> = this._onDidChangeCurrentThread.event;

	private readonly _onDidChangeStreamState = new Emitter<{ threadId: string }>();
	readonly onDidChangeStreamState: Event<{ threadId: string }> = this._onDidChangeStreamState.event;

	// dedicated event for queue changes — fires with the threadId that changed
	private readonly _onDidChangeMessageQueue = new Emitter<{ threadId: string }>();
	readonly onDidChangeMessageQueue: Event<{ threadId: string }> = this._onDidChangeMessageQueue.event;

	readonly streamState: ThreadStreamState = {}
	state: ThreadsState // allThreads is persisted, currentThread is not

	// Message queue: stores pending messages per thread
	private messageQueue: { [threadId: string]: Array<{ userMessage: string, selections?: StagingSelectionItem[], images?: ImageAttachment[] }> } = {}

	// Task planning: stores task plans per thread
	private taskPlans: { [threadId: string]: TaskPlan[] } = {}

	// SessionStart hook: fired once per app session. Source is 'resume' when the
	// constructor loaded stored threads, else 'startup'. `_sessionStartFired`
	// guards against re-firing across thread switches within one session.
	private _sessionStartSource: 'startup' | 'resume' = 'startup'
	private _sessionStartFired = false

	// PERFORMANCE: Debounce timer for storage
	private _storeThreadsDebounceTimer: any = null;
	private readonly MAX_THREADS_IN_STORAGE = 5;
	private _cleanupInterval: any = null;

	// used in checkpointing
	// private readonly _userModifiedFilesToCheckInCheckpoints = new LRUCache<string, null>(50)



	constructor(
		@IStorageService private readonly _storageService: IStorageService,
		@IVoidModelService private readonly _voidModelService: IVoidModelService,
		@ILLMMessageService private readonly _llmMessageService: ILLMMessageService,
		@IToolsService private readonly _toolsService: IToolsService,
		@IVoidSettingsService private readonly _settingsService: IVoidSettingsService,
		@ILanguageFeaturesService private readonly _languageFeaturesService: ILanguageFeaturesService,
		@IMetricsService private readonly _metricsService: IMetricsService,
		@IEditCodeService private readonly _editCodeService: IEditCodeService,
		@INotificationService private readonly _notificationService: INotificationService,
		@IConvertToLLMMessageService private readonly _convertToLLMMessagesService: IConvertToLLMMessageService,
		@IWorkspaceContextService private readonly _workspaceContextService: IWorkspaceContextService,
		@IDirectoryStrService private readonly _directoryStringService: IDirectoryStrService,
		@IFileService private readonly _fileService: IFileService,
		@IMCPService private readonly _mcpService: IMCPService,
		@IACPService private readonly _acpService: IACPService,
		@IComposioService private readonly _composioService: IComposioService,
		@IVisionService private readonly _visionService: IVisionService,
		@IModelService private readonly _modelService: IModelService,
		@IToolOrchestrationService private readonly _orchestrationService: IToolOrchestrationService,
		@IHookService private readonly _hookService: IHookService,
		@ISubagentService private readonly _subagentService: ISubagentService,
	) {
		super()
		this.state = { allThreads: {}, currentThreadId: null as unknown as string } // default state

		const readThreads = this._readAllThreads() || {}

		// If stored threads were loaded, this is a resume; otherwise a fresh startup.
		this._sessionStartSource = (Object.keys(readThreads).length > 0) ? 'resume' : 'startup'

		const allThreads = this._ensureThreadStateDefaults(readThreads)
		this.state = {
			allThreads: allThreads,
			currentThreadId: null as unknown as string, // gets set in startNewThread()
		}

		// always be in a thread
		this.openNewThread()


		// keep track of user-modified files
		const disposablesOfModelId: { [modelId: string]: IDisposable[] } = {}
		this._register(
			this._modelService.onModelAdded(e => {
				const uri = e.uri
				if (!(uri.toString() in disposablesOfModelId)) disposablesOfModelId[uri.toString()] = []
				disposablesOfModelId[uri.toString()].push(
					e.onDidChangeContent(() => {
						const threadId = this.state.currentThreadId
						const thread = this.state.allThreads[threadId]
						if (thread) {
							thread.filesWithUserChanges.add(uri.fsPath)
						}
					})
				)
			})
		)
		this._register(this._modelService.onModelRemoved(e => {
			const uri = e.uri
			if (!(uri.toString() in disposablesOfModelId)) return
			disposablesOfModelId[uri.toString()].forEach(d => d.dispose())
			delete disposablesOfModelId[uri.toString()]
		}))

	}

	private _clearUserChanges(threadId: string) {
		const thread = this.state.allThreads[threadId]
		if (thread) {
			thread.filesWithUserChanges.clear()
		}
	}

	override dispose() {
		if (this._storeThreadsDebounceTimer) {
			clearTimeout(this._storeThreadsDebounceTimer);
			this._storeAllThreadsNow(this.state.allThreads);
		}
		if (this._cleanupInterval) {
			clearInterval(this._cleanupInterval);
			this._cleanupInterval = null;
		}
		super.dispose();
	}

	async focusCurrentChat(timeout: number = 5000): Promise<void> {
		const threadId = this.state.currentThreadId
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		// Wait for mountedInfo with timeout to prevent hanging
		const mountedInfo = thread.state.mountedInfo
		if (!mountedInfo?.whenMounted) return

		// Race between mount and timeout
		try {
			const s = await Promise.race([
				mountedInfo.whenMounted,
				new Promise<WhenMounted | null>((_, reject) =>
					setTimeout(() => reject(new Error('focusCurrentChat timeout')), timeout)
				)
			])
			if (!this.isCurrentlyFocusingMessage()) {
				s?.textAreaRef.current?.focus()
			}
		} catch {
			// Timeout - component may not have mounted yet, just skip focusing
			voidDevLog('[chatThreadService] focusCurrentChat timed out waiting for mount')
		}
	}
	async blurCurrentChat() {
		const threadId = this.state.currentThreadId
		const thread = this.state.allThreads[threadId]
		if (!thread) return
		const s = await thread.state.mountedInfo?.whenMounted
		if (!this.isCurrentlyFocusingMessage()) {
			s?.textAreaRef.current?.blur()
		}
	}



	dangerousSetState = (newState: ThreadsState) => {
		this.state = newState
		this._onDidChangeCurrentThread.fire()
	}
	resetState = () => {
		// MEMORY FIX: Clean up all auxiliary data when resetting state
		this.toolCallHistory = {};
		this.toolResultCache = {};
		this.messageQueue = {};
		this.taskPlans = {};

		this.state = { allThreads: {}, currentThreadId: null as unknown as string } // see constructor
		this.openNewThread()
		this._onDidChangeCurrentThread.fire()
	}

	// !!! this is important for properly restoring URIs from storage
	// should probably re-use code from void/src/vs/base/common/marshalling.ts instead. but this is simple enough
	private _convertThreadDataFromStorage(threadsStr: string): ChatThreads {
		return JSON.parse(threadsStr, (key, value) => {
			if (value && typeof value === 'object' && value.$mid === 1) { // $mid is the MarshalledId. $mid === 1 means it is a URI
				return URI.revive(value);
			}
			return value;
		});
	}

	/**
	 * Migrate thread data from older storage versions
	 * Add migration logic here when changing data structures
	 */
	private _migrateThreads(threads: ChatThreads, fromVersion: number): ChatThreads {
		let migratedThreads = threads;

		// Version 1 migrations (current version)
		if (fromVersion < 1) {
			voidDevLog(`[Migration] Migrating threads from version ${fromVersion} to version 1`);

			// Migration: Ensure all messages have required fields
			// Migration: Convert old Set serialization to arrays
			for (const [, thread] of Object.entries(migratedThreads)) {
				if (thread && 'filesWithUserChanges' in thread) {
					// Convert Set to array if serialized as object
					if (!(thread.filesWithUserChanges instanceof Set)) {
						const raw = thread.filesWithUserChanges as unknown as string[] | Record<string, string> | null;
						if (Array.isArray(raw)) {
							thread.filesWithUserChanges = new Set(raw);
						} else if (typeof raw === 'object' && raw !== null) {
							// Handle object serialization of Set
							thread.filesWithUserChanges = new Set(Object.values(raw));
						}
					}
				}
			}

			voidDevLog(`[Migration] Migration complete. Processed ${Object.keys(migratedThreads).length} threads`);
		}

		return migratedThreads;
	}

	/**
	 * Get storage version from storage service
	 */
	private _getStorageVersion(): number {
		const versionStr = this._storageService.get(THREAD_STORAGE_VERSION_KEY, StorageScope.APPLICATION);
		if (!versionStr) return 0; // No version means old data (pre-versioning)
		return parseInt(versionStr, 10) || 0;
	}

	/**
	 * Save storage version to storage service
	 */
	private _saveStorageVersion(version: number): void {
		this._storageService.store(
			THREAD_STORAGE_VERSION_KEY,
			version.toString(),
			StorageScope.APPLICATION,
			StorageTarget.USER
		);
	}

	private _readAllThreads(): ChatThreads | null {
		// MIGRATION: older versions stored every thread in one JSON blob. Fan the
		// blob out into per-thread keys and delete it, so saves become
		// O(changed thread) instead of O(all threads).
		const legacyStr = this._storageService.get(THREAD_STORAGE_KEY, StorageScope.APPLICATION);
		if (legacyStr) {
			try {
				const legacyThreads = this._convertThreadDataFromStorage(legacyStr);
				for (const [threadId, thread] of Object.entries(legacyThreads)) {
					if (!thread) continue;
					const json = this._serializeThreadForStorage(threadId, thread);
					this._storageService.store(THREAD_STORAGE_KEY_PREFIX + threadId, json, StorageScope.APPLICATION, StorageTarget.USER);
					this._persistedThreadJson.set(threadId, json);
				}
				voidDevLog(`[Storage] Migrated ${Object.keys(legacyThreads).length} thread(s) from legacy blob to per-thread storage`);
			} catch (e) {
				voidDevWarn('[Storage] Failed to migrate legacy thread blob:', e);
			}
			this._storageService.remove(THREAD_STORAGE_KEY, StorageScope.APPLICATION);
		}

		// Per-thread storage: enumerate keys and reassemble the map.
		const perThreadKeys = this._storageService.keys(StorageScope.APPLICATION, StorageTarget.USER)
			.filter(key => key.startsWith(THREAD_STORAGE_KEY_PREFIX));
		if (perThreadKeys.length === 0) return null;

		const threads: ChatThreads = {};
		for (const key of perThreadKeys) {
			const threadId = key.slice(THREAD_STORAGE_KEY_PREFIX.length);
			const threadsStr = this._storageService.get(key, StorageScope.APPLICATION);
			if (!threadsStr) continue;
			try {
				const single = this._convertThreadDataFromStorage(threadsStr);
				const thread = single[threadId];
				if (thread) threads[threadId] = thread;
			} catch (e) {
				voidDevWarn(`[Storage] Failed to parse stored thread ${threadId}:`, e);
			}
		}

		// Apply data-schema migrations if needed (unchanged semantics from the
		// legacy blob era — this migrates thread data shapes, not storage layout).
		const storedVersion = this._getStorageVersion();
		if (storedVersion < CURRENT_THREAD_STORAGE_VERSION) {
			voidDevLog(`[Storage] Found threads from version ${storedVersion}, current is ${CURRENT_THREAD_STORAGE_VERSION}`);
			const migratedThreads = this._migrateThreads(threads, storedVersion);
			// Save migrated data
			this._storeAllThreads(migratedThreads);
			this._saveStorageVersion(CURRENT_THREAD_STORAGE_VERSION);
			return migratedThreads;
		}

		return threads;
	}

	/**
	 * Serialize a single thread for storage. Wrapped as `{ [threadId]: thread }`
	 * so `_convertThreadDataFromStorage` (with its URI revival) round-trips it.
	 * Does not mutate the live thread object.
	 */
	private _serializeThreadForStorage(threadId: string, thread: ThreadType): string {
		const { filesWithUserChanges, ...rest } = thread;
		const serializable = {
			...rest,
			filesWithUserChanges: filesWithUserChanges instanceof Set ? Array.from(filesWithUserChanges) : filesWithUserChanges,
		};
		return JSON.stringify({ [threadId]: serializable });
	}

	private _storeAllThreads(threads: ChatThreads) {
		if (this._storeThreadsDebounceTimer) {
			clearTimeout(this._storeThreadsDebounceTimer);
		}

		this._storeThreadsDebounceTimer = setTimeout(() => {
			this._storeAllThreadsNow(threads);
			this._storeThreadsDebounceTimer = null;
		}, 1000); // 1 second debounce
	}

	private _storeAllThreadsNow(threads: ChatThreads) {
		const normalizedThreads = this._ensureThreadStateDefaults(threads)

		// PERFORMANCE: Prune old threads to prevent storage bloat and high memory usage
		const sortedThreadIds = Object.keys(normalizedThreads).sort((a, b) => {
			const timeA = new Date(normalizedThreads[a]?.lastModified ?? 0).getTime();
			const timeB = new Date(normalizedThreads[b]?.lastModified ?? 0).getTime();
			return timeB - timeA; // Descending order (newest first)
		});

		// Keep the most recent threads in storage
		const threadIdsToKeep = new Set(sortedThreadIds.slice(0, this.MAX_THREADS_IN_STORAGE));

		// MEMORY FIX: Clean up auxiliary data for pruned threads
		const threadIdsToPrune = sortedThreadIds.slice(this.MAX_THREADS_IN_STORAGE);
		for (const threadId of threadIdsToPrune) {
			delete this.toolCallHistory[threadId];
			delete this.toolResultCache[threadId];
			delete this.messageQueue[threadId];
			delete this.taskPlans[threadId];

			// ALSO: If we are pruning the current thread (unlikely but possible), clear it
			if (this.state.currentThreadId === threadId) {
				this.openNewThread();
			}
		}

		// PER-THREAD PERSISTENCE (ported from a-coder-cli's JSONL approach): each
		// thread lives under its own storage key, and a save only serializes threads
		// whose serialized content actually changed — usually exactly one — instead
		// of stringify-ing the entire stored history on every save.
		for (const threadId of threadIdsToKeep) {
			const thread = normalizedThreads[threadId];
			if (!thread) continue;
			const json = this._serializeThreadForStorage(threadId, thread);
			if (this._persistedThreadJson.get(threadId) === json) continue;
			this._storageService.store(THREAD_STORAGE_KEY_PREFIX + threadId, json, StorageScope.APPLICATION, StorageTarget.USER);
			this._persistedThreadJson.set(threadId, json);
		}

		// Threads that were deleted or pruned: drop their storage keys.
		for (const threadId of Array.from(this._persistedThreadJson.keys())) {
			if (!threadIdsToKeep.has(threadId)) {
				this._storageService.remove(THREAD_STORAGE_KEY_PREFIX + threadId, StorageScope.APPLICATION);
				this._persistedThreadJson.delete(threadId);
			}
		}
	}

	private _ensureThreadStateDefaults(threads: ChatThreads): ChatThreads {
		const nextThreads: ChatThreads = {}
		for (const [threadId, thread] of Object.entries(threads)) {
			if (!thread) {
				nextThreads[threadId] = thread
				continue
			}
			nextThreads[threadId] = {
				...thread,
				filesWithUserChanges: thread.filesWithUserChanges instanceof Set
					? thread.filesWithUserChanges
					: new Set(Array.isArray(thread.filesWithUserChanges) ? thread.filesWithUserChanges : []),
				state: {
					...thread.state,
					autoContinueEnabled: thread.state?.autoContinueEnabled ?? false,
					activeWorkflow: thread.state?.activeWorkflow ?? null,
					queueBehavior: thread.state?.queueBehavior ?? 'wait_for_workflow',
					voiceModeActive: thread.state?.voiceModeActive ?? false,
				},
			}
		}
		return nextThreads
	}


	// this should be the only place this.state = ... appears besides constructor
	private _setState(state: Partial<ThreadsState>, doNotRefreshMountInfo?: boolean) {
		// Plans are stored per-thread; whenever the active thread changes, switch
		// the planning services' active context so the UI and tool calls reflect it.
		if (state.currentThreadId !== undefined && state.currentThreadId !== this.state.currentThreadId) {
			this._toolsService.getPlanningService().switchToThread(state.currentThreadId)
			this._toolsService.getImplementationPlanningService().switchToThread(state.currentThreadId)
		}

		const newState = {
			...this.state,
			...state
		}

		this.state = newState

		this._onDidChangeCurrentThread.fire()


		// if we just switched to a thread, update its current stream state if it's not streaming to possibly streaming
		const threadId = newState.currentThreadId
		const streamState = this.streamState[threadId]
		if (streamState?.isRunning === undefined && !streamState?.error) {

			// set streamState
			const messages = newState.allThreads[threadId]?.messages
			const lastMessage = messages && messages[messages.length - 1]
			// if awaiting user but stream state doesn't indicate it (happens if restart Void)
			if (lastMessage && lastMessage.role === 'tool' && lastMessage.type === 'tool_request')
				this._setStreamState(threadId, { isRunning: 'awaiting_user', })

			// if running now but stream state doesn't indicate it (happens if restart Void), cancel that last tool
			if (lastMessage && lastMessage.role === 'tool' && lastMessage.type === 'running_now') {

				this._updateLatestTool(threadId, { role: 'tool', type: 'rejected', content: lastMessage.content, id: lastMessage.id, rawParams: lastMessage.rawParams, result: null, name: lastMessage.name, params: lastMessage.params, mcpServerName: lastMessage.mcpServerName })
			}

		}


		// if we did not just set the state to true, set mount info
		if (doNotRefreshMountInfo) return

		let whenMountedResolver: (w: WhenMounted) => void
		const whenMountedPromise = new Promise<WhenMounted>((res) => whenMountedResolver = res)

		this._setThreadState(threadId, {
			mountedInfo: {
				whenMounted: whenMountedPromise,
				mountedIsResolvedRef: { current: false },
				_whenMountedResolver: (w: WhenMounted) => {
					whenMountedResolver(w)
					const mountInfo = this.state.allThreads[threadId]?.state.mountedInfo
					if (mountInfo) mountInfo.mountedIsResolvedRef.current = true
				},
			}
		}, true) // do not trigger an update



	}


	private _setStreamState(threadId: string, state: ThreadStreamState[string]) {
		// Preserve tokenUsage when updating state
		const currentTokenUsage = this.streamState[threadId]?.tokenUsage;
		if (state && currentTokenUsage) {
			state.tokenUsage = currentTokenUsage;
		}
		this.streamState[threadId] = state
		this._onDidChangeStreamState.fire({ threadId })

		// MEMORY OPTIMIZATION: If state is undefined, delete the key to free memory
		// This ensures the llmInfo with large strings is garbage collected
		if (state === undefined) {
			delete this.streamState[threadId];
		}
	}


	// ---------- streaming ----------



	private _currentModelSelectionProps = () => {
		// these settings should not change throughout the loop (eg anthropic breaks if you change its thinking mode and it's using tools)
		const featureName: FeatureName = 'Chat'
		const modelSelection = this._settingsService.state.modelSelectionOfFeature[featureName]
		const modelSelectionOptions = modelSelection ? this._settingsService.state.optionsOfModelSelection[featureName][modelSelection.providerName]?.[modelSelection.modelName] : undefined
		return { modelSelection, modelSelectionOptions }
	}



	private _swapOutLatestStreamingToolWithResult = (threadId: string, tool: ChatMessage & { role: 'tool' }) => {
		const messages = this.state.allThreads[threadId]?.messages
		if (!messages) return false
		const lastMsg = messages[messages.length - 1]
		if (!lastMsg) return false

		if (lastMsg.role === 'tool' && lastMsg.type !== 'invalid_params') {
			this._editMessageInThread(threadId, messages.length - 1, tool)
			return true
		}
		return false
	}

private _updateLatestTool = (threadId: string, tool: ChatMessage & { role: 'tool' }) => {
		const swapped = this._swapOutLatestStreamingToolWithResult(threadId, tool)
		if (swapped) return
		this._addMessageToThread(threadId, tool)
	}

	// Update or add a tool message. In sequential mode we swap "the latest" tool
	// message in place, so each call evolves running_now -> success as ONE message.
	//
	// In parallel mode several tools mutate the same thread concurrently, so we
	// can't swap "the latest" message — that would let one tool clobber another's
	// in-progress row. Instead we update THIS tool's own row in place by id,
	// appending only the first time we see the id. This keeps the one-evolving-
	// message-per-tool invariant that both downstream consumers rely on:
	//   - convertToLLMMessageService._chatMessagesToSimpleMessages emits every
	//     role:'tool' message without deduping by id, so a leftover running_now
	//     next to the success would send the provider two results for one
	//     tool_call_id (OpenAI-style APIs reject duplicate tool_call_ids).
	//   - NestedToolGroup counts one card per message, so a leftover running_now
	//     would double the count ("2N tools ran in parallel"), spin the spinner
	//     forever, and render each tool twice.
	private _updateToolMessage = (threadId: string, tool: ChatMessage & { role: 'tool' }, parallelMode: boolean) => {
		if (parallelMode) {
			const messages = this.state.allThreads[threadId]?.messages
			if (messages) {
				const existingIdx = findLastIdx(messages, m => m.role === 'tool' && m.id === tool.id)
				if (existingIdx !== -1) {
					this._editMessageInThread(threadId, existingIdx, tool)
					return
				}
			}
			this._addMessageToThread(threadId, tool)
		} else {
			this._updateLatestTool(threadId, tool)
		}
	}

	approveLatestToolRequest(threadId: string, toolId?: string) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return // should never happen

		let toolMsgIdx = -1;
		if (toolId) {
			toolMsgIdx = findLastIdx(thread.messages, m => m.role === 'tool' && m.type === 'tool_request' && m.id === toolId);
		} else {
			toolMsgIdx = findLastIdx(thread.messages, m => m.role === 'tool' && m.type === 'tool_request');
		}

		if (toolMsgIdx === -1) return;

		const toolMsg = thread.messages[toolMsgIdx] as ToolMessage<ToolName> & { type: 'tool_request' };

		this._wrapRunAgentToNotify(
			this._runChatAgent({ callThisToolFirst: toolMsg, threadId, ...this._currentModelSelectionProps() })
			, threadId
		)
	}
	rejectLatestToolRequest(threadId: string, toolId?: string) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return // should never happen

		let toolMsgIdx = -1;
		if (toolId) {
			toolMsgIdx = findLastIdx(thread.messages, m => m.role === 'tool' && m.type === 'tool_request' && m.id === toolId);
		} else {
			toolMsgIdx = findLastIdx(thread.messages, m => m.role === 'tool' && m.type === 'tool_request');
		}

		if (toolMsgIdx === -1) return;

		const toolMsg = thread.messages[toolMsgIdx] as ToolMessage<ToolName> & { type: 'tool_request' };

		let params: ToolCallParams<ToolName> = toolMsg.params

		const { name, id, rawParams, mcpServerName } = toolMsg

		const errorMessage = this.toolErrMsgs.rejected
		this._updateLatestTool(threadId, { role: 'tool', type: 'rejected', params: params, name: name, content: errorMessage, result: null, id, rawParams, mcpServerName })
		this._setStreamState(threadId, undefined)
	}

	skipLatestToolRequest(threadId: string, toolId?: string) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return // should never happen

		let toolMsgIdx = -1;
		if (toolId) {
			toolMsgIdx = findLastIdx(thread.messages, m => m.role === 'tool' && m.type === 'tool_request' && m.id === toolId);
		} else {
			toolMsgIdx = findLastIdx(thread.messages, m => m.role === 'tool' && m.type === 'tool_request');
		}

		if (toolMsgIdx === -1) return;

		const toolMsg = thread.messages[toolMsgIdx] as ToolMessage<ToolName> & { type: 'tool_request' };

		let params: ToolCallParams<ToolName> = toolMsg.params

		const { name, id, rawParams, mcpServerName } = toolMsg

		// Mark as skipped (similar to rejected but with different message)
		const skipMessage = 'Tool skipped by user - continuing with next action'
		this._updateLatestTool(threadId, { role: 'tool', type: 'rejected', params: params, name: name, content: skipMessage, result: null, id, rawParams, mcpServerName })

		// Continue the agent loop instead of stopping
		this._wrapRunAgentToNotify(
			this._runChatAgent({ threadId, ...this._currentModelSelectionProps() })
			, threadId
		)
	}

	submitToolResult(threadId: string, toolId: string, result: any) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return // should never happen

		// Find the tool request message
		const toolMsgIdx = findLastIdx(thread.messages, m => m.role === 'tool' && m.type === 'tool_request' && m.id === toolId);
		if (toolMsgIdx === -1) {
			voidDevWarn(`[chatThreadService] submitToolResult: Tool request not found for toolId ${toolId}`);
			return;
		}

		const toolMsg = thread.messages[toolMsgIdx] as ToolMessage<ToolName> & { type: 'tool_request' };
		const { name, rawParams, mcpServerName, thought_signature } = toolMsg;

		voidDevLog(`[chatThreadService] submitToolResult: Submitting result for tool ${name} (id: ${toolId})`);

		// Format the result as a string for the LLM
		let toolResultStr: string;
		try {
			toolResultStr = JSON.stringify(result, null, 2);
		} catch (error) {
			toolResultStr = String(result);
		}

		voidDevLog(`[chatThreadService] submitToolResult: Result: ${toolResultStr.substring(0, 200)}${toolResultStr.length > 200 ? '...' : ''}`);

		// Update the tool message with the result (type: 'success')
		this._updateLatestTool(threadId, {
			role: 'tool',
			type: 'success',
			name: name,
			params: toolMsg.params,
			result: result,
			content: toolResultStr,
			id: toolId,
			rawParams: rawParams,
			mcpServerName: mcpServerName,
			thought_signature: thought_signature
		});

		// Resume the agent directly (without re-executing the tool)
		// We use _runChatAgent without callThisToolFirst so it just continues
		// processing with the tool result we just added to the thread
		voidDevLog(`[chatThreadService] submitToolResult: Resuming agent for thread ${threadId}`);
		this._wrapRunAgentToNotify(
			this._runChatAgent({ threadId, ...this._currentModelSelectionProps() })
			, threadId
		)
	}

	private _computeMCPServerOfToolName = (toolName: string) => {
		const isComposio = this._composioService.isComposioTool(toolName);
		if (isComposio) {
			return 'composio_tool_router'
		}
		const acpTool = this._acpService.getACPAgents()?.find(t => t.name === toolName);
		if (acpTool) {
			return 'acp_agent_router'
		}
		return this._mcpService.getMCPTools()?.find(t => t.name === toolName)?.mcpServerName
	}

	async abortRunning(threadId: string) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return // should never happen

		// add assistant message
		if (this.streamState[threadId]?.isRunning === 'LLM') {
			const { displayContentSoFar, reasoningSoFar, toolCallsSoFar } = this.streamState[threadId].llmInfo
			this._addMessageToThread(threadId, { role: 'assistant', displayContent: displayContentSoFar, reasoning: reasoningSoFar, anthropicReasoning: null })
			if (toolCallsSoFar) {
				for (const tc of toolCallsSoFar) {
					this._addMessageToThread(threadId, { role: 'interrupted_streaming_tool', name: tc.name, mcpServerName: this._computeMCPServerOfToolName(tc.name) })
				}
			}
		}
		// add tool that's running
		else if (this.streamState[threadId]?.isRunning === 'tool') {
			const { toolName, toolParams, id, content: content_, rawParams, mcpServerName } = this.streamState[threadId].toolInfo
			const content = content_ || this.toolErrMsgs.interrupted
			this._updateLatestTool(threadId, { role: 'tool', name: toolName, params: toolParams, id, content, rawParams, type: 'rejected', result: null, mcpServerName })
		}
		// reject the tool for the user if relevant
		else if (this.streamState[threadId]?.isRunning === 'awaiting_user') {
			this.rejectLatestToolRequest(threadId)
		}
		else if (this.streamState[threadId]?.isRunning === 'idle') {
			// do nothing
		}

		// interrupt any effects
		const interrupt = await this.streamState[threadId]?.interrupt
		if (typeof interrupt === 'function')
			interrupt()

		// Cancel any subagents spawned by this thread so hitting Stop in chat
		// also stops background/foreground delegated work.
		this._subagentService.cancelAllForThread(threadId)

		this._setStreamState(threadId, undefined)
	}



	// TOON service for compressing large tool results
	private readonly _toonService = new ToonService();

	// USAGE ANCHORING: last real prompt usage per thread, from the most recent
	// successful LLM call. Lets prepareLLMChatMessages anchor on measured tokens
	// and only estimate trailing messages instead of re-tokenizing everything.
	// Ephemeral (not persisted) — a cold start falls back to a full count.
	private readonly usageAnchorByThreadId = new Map<string, UsageAnchor>();

	// PER-THREAD PERSISTENCE: last-persisted serialized JSON per thread, used to
	// skip re-writing threads whose content did not change since the last save.
	private readonly _persistedThreadJson = new Map<string, string>();

	// AGENT LOOP EVENTS: typed, observational events for loop transitions (see
	// agentLoopEvents.ts). Emission never throws into the loop.
	private readonly _onAgentLoopEvent = new Emitter<AgentLoopEvent>();
	readonly onAgentLoopEvent = this._onAgentLoopEvent.event;
	private _emitAgentEvent(event: AgentLoopEvent): void {
		try {
			this._onAgentLoopEvent.fire(event);
		} catch (e) {
			voidDevWarn('[chatThreadService] agent loop event subscriber threw:', e)
		}
	}

	private readonly toolErrMsgs = {
		rejected: 'Tool call was rejected by the user.',
		interrupted: 'Tool call was interrupted by the user.',
		errWhenStringifying: (error: any) => `Tool call succeeded, but there was an error stringifying the output.\n${getErrorMessage(error)}`
	}


	// private readonly _currentlyRunningToolInterruptor: { [threadId: string]: (() => void) | undefined } = {}


	// Track tool call history for loop detection
	private toolCallHistory: { [threadId: string]: Array<{ name: string, params: any, result: any, type: string, _paramsKey?: string }> } = {};

	// Per-thread memoization of read-only tool results (read_file, search_*, etc.)
	// keyed by a stable, complete param signature. Invalidated wholesale on any
	// mutating/external tool call (see _runToolCall).
	private toolResultCache: { [threadId: string]: Map<string, { resultStr: string, result: ToolResult<ToolName> }> } = {};

	private _truncateToolResult(result: any): any {
		// TWO-LIMIT TRUNCATION (ported from a-coder-cli): independent line + byte
		// limits, whichever is hit first, always cutting on a line boundary so
		// line-number references stay intact. The old substring() cut could split
		// a line mid-row and corrupt line-numbered output (read_file etc.).
		if (typeof result === 'string') {
			return truncateToolOutputWithNotice(result, { maxBytes: MAX_TOOL_RESULT_LENGTH });
		}

		if (result && typeof result === 'object') {
			// If it's a ToolResult object (e.g. from read_file)
			if ('content' in result && typeof result.content === 'string') {
				return {
					...result,
					content: truncateToolOutputWithNotice(result.content, { maxBytes: MAX_TOOL_RESULT_LENGTH })
				};
			}

			// Handle MCP tool results which often have a 'content' array
			if ('content' in result && Array.isArray(result.content)) {
				return {
					...result,
					content: result.content.map((item: any) => {
						if (item && typeof item === 'object' && 'text' in item && typeof item.text === 'string') {
							return { ...item, text: truncateToolOutputWithNotice(item.text, { maxBytes: MAX_TOOL_RESULT_LENGTH }) };
						}
						return item;
					})
				};
			}
		}

		return result;
	}

	// MEMORY OPTIMIZATION: Fast param comparison without JSON.stringify
	// Creates a hash key for params to avoid expensive stringify operations
	private _getToolParamsKey(toolName: string, params: any): string {
		if (!params) return toolName;
		// Fast path: extract key identifying fields instead of full stringify
		if (typeof params === 'object') {
			// For common tool params, use specific key fields
			if (params.uri?.fsPath) {
				return `${toolName}:${params.uri.fsPath}`;
			}
			if (params.query) {
				return `${toolName}:${params.query}`;
			}
			if (params.command) {
				return `${toolName}:${params.command}`;
			}
			// Fallback: use tool name + first few keys
			const keyParts = Object.keys(params).slice(0, 3).map(k => {
				const v = params[k];
				if (typeof v === 'string') return `${k}=${v.slice(0, 50)}`;
				if (typeof v === 'number') return `${k}=${v}`;
				if (v?.fsPath) return `${k}=${v.fsPath}`;
				return `${k}=obj`;
			});
			return `${toolName}:${keyParts.join(',')}`;
		}
		return `${toolName}:${String(params)}`;
	}

	// Stable, complete cache key for tool results. Unlike _getToolParamsKey (a
	// lossy fast-path for loop detection), this captures every param field so
	// e.g. read_file with different line ranges can't collide.
	private _toolCacheKey(toolName: string, params: object): string {
		const stableStringify = (value: unknown): string => {
			if (value === null || value === undefined) return 'null'
			if (typeof value !== 'object') return String(value)
			// URI-like values: identify by their filesystem path.
			if ('fsPath' in value && typeof (value as { fsPath: unknown }).fsPath === 'string') return (value as { fsPath: string }).fsPath
			if (Array.isArray(value)) return '[' + value.map(stableStringify).join(',') + ']'
			const keys = Object.keys(value).sort()
			return '{' + keys.map(k => `${k}:${stableStringify((value as Record<string, unknown>)[k])}`).join(',') + '}'
		}
		return `${toolName}:${stableStringify(params)}`
	}

	// Predictive progress messages based on tool name
	private getPredictiveProgressMessage(toolName: string, params: any): string {
		switch (toolName) {
			case 'ls_dir': return `Exploring directory: ${params.uri?.fsPath || '...'}`;
			case 'read_file': return `Reading file: ${params.uri?.fsPath || '...'}`;
			case 'search_for_files': return `Searching codebase for: "${params.query}"`;
			case 'run_command': return `Executing command: "${params.command}"`;
		case 'edit_file': return `Applying edits to: ${params.uri?.fsPath || '...'}`;
		case 'edit_files': return `Applying edits to ${params.edits?.length || 0} file(s)...`;
		case 'rewrite_file': return `Rewriting file: ${params.uri?.fsPath || '...'}`;
			case 'get_dir_tree': return `Analyzing project structure...`;
			case 'search_pathnames_only': return `Locating files matching: "${params.query}"`;
			case 'create_todo': return `Creating todo list...`;
			case 'fast_context': return `Morph: Searching for "${params.query}"...`;
			default: return `Executing ${toolName}...`;
		}
	}

	/** Snapshot of the terminal command auto-approval policy for the logic module. */
	private _terminalAutoApproveSettings(): TerminalAutoApproveSettings {
		const g = this._settingsService.state.globalSettings
		return {
			masterToggle: g.autoApprove['terminal'],
			terminalAllowPatterns: g.terminalAllowPatterns ?? [],
			terminalDenyPatterns: g.terminalDenyPatterns ?? [],
			terminalReadOnlyAutoApprove: !!g.terminalReadOnlyAutoApprove,
		}
	}

	// returns true when the tool call is waiting for user approval
	private _runToolCall = async (
		threadId: string,
		toolName: ToolName,
		toolId: string,
		mcpServerName: string | undefined,
		opts: { preapproved: true, unvalidatedToolParams: RawToolParamsObj, validatedParams: ToolCallParams<ToolName>, thought_signature?: string } | { preapproved: false, unvalidatedToolParams: RawToolParamsObj, thought_signature?: string },
		parallelMode?: boolean,
		parallelBatchId?: string,
	): Promise<{ awaitingUserApproval?: boolean, interrupted?: boolean }> => {

		// ... internal vars ...
		let toolParams: ToolCallParams<ToolName>
		let toolResult: ToolResult<ToolName>
		let toolResultStr: string

		const isBuiltInTool = isABuiltinToolName(toolName)

		if (!opts.preapproved) {
			try {
				if (isBuiltInTool) {
					const params = this._toolsService.validateParams[toolName](opts.unvalidatedToolParams)
					toolParams = params
				}
				else {
					toolParams = opts.unvalidatedToolParams
				}
			}
			catch (error) {
				const errorMessage = getErrorMessage(error)
				this._addMessageToThread(threadId, { role: 'tool', type: 'invalid_params', rawParams: opts.unvalidatedToolParams, result: null, name: toolName, content: errorMessage, id: toolId, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId })
				return {}
			}

			// PLAN-MODE BACKSTOP — block mutating/external tools at the dispatch layer.
			// Plan mode is research-only. The prompt-level tool omission in prompts.ts is
			// not a guarantee (external MCP/Composio/ACP tools are advertised in plan mode
			// and are opaque to us), so this guard catches writes and external tools that
			// leak in regardless of how they were advertised — before the approval gate
			// shows a tool_request, and before any edit checkpoint is recorded.
			if (this._settingsService.state.globalSettings.chatMode === 'plan') {
				const planApprovalType = isBuiltInTool ? approvalTypeOfBuiltinToolName[toolName] : 'MCP tools'
				const planBlocked = !isBuiltInTool
					|| planApprovalType === 'edits'
					|| planApprovalType === 'terminal'
					|| planApprovalType === 'code execution'
					|| planApprovalType === 'image generation'
					|| planApprovalType === 'repo'
				if (planBlocked) {
					const blockReason = `Tool '${toolName}' is blocked in Plan mode. Switch to Code or Agent mode to make changes.`
					this._addMessageToThread(threadId, { role: 'tool', type: 'tool_error', params: toolParams, result: blockReason, name: toolName, content: blockReason, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId })
					return {}
				}
			}

			// LOOP DETECTION: Check if we've tried this exact failing call recently
			const history = this.toolCallHistory[threadId] || [];
			const lastCall = history[history.length - 1];
			// MEMORY OPTIMIZATION: Use fast key comparison instead of JSON.stringify
			const currentParamsKey = this._getToolParamsKey(toolName, toolParams);
			if (lastCall && lastCall.name === toolName && lastCall.type !== 'success') {
				const lastParamsKey = lastCall._paramsKey || this._getToolParamsKey(lastCall.name, lastCall.params);
				if (lastParamsKey === currentParamsKey) {
					// We are repeating a failing call. Add a note to help the agent break out.
					voidDevWarn(`[chatThreadService] Loop detected for tool ${toolName}.`);
					// We don't block it here, but we will ensure the result contains a hint for the agent.
				}
			}

			if (toolName === 'edit_file') { this._addToolEditCheckpoint({ threadId, uri: (toolParams as BuiltinToolCallParams['edit_file']).uri }) }
			if (toolName === 'edit_files') {
				for (const edit of (toolParams as BuiltinToolCallParams['edit_files']).edits) {
					this._addToolEditCheckpoint({ threadId, uri: edit.uri })
				}
			}
			if (toolName === 'rewrite_file') { this._addToolEditCheckpoint({ threadId, uri: (toolParams as BuiltinToolCallParams['rewrite_file']).uri }) }

			const approvalType = isBuiltInTool ? approvalTypeOfBuiltinToolName[toolName] : 'MCP tools'
			if (approvalType) {
				// Terminal tools use the layered command-allowlist policy (deny list,
				// master toggle, allow list, read-only preset); all other approval
				// types use the simple per-type boolean.
				const autoApprove = approvalType === 'terminal'
					? shouldAutoApproveTerminalTool(toolName, toolName === 'run_command' ? (toolParams as BuiltinToolCallParams['run_command']).command : undefined, this._terminalAutoApproveSettings())
					: !!this._settingsService.state.globalSettings.autoApprove[approvalType]
				const content = toolName === 'render_form' || toolName === 'create_quiz' ? 'Please complete the interactive content below.' : '(Awaiting user permission...)'
				this._addMessageToThread(threadId, { role: 'tool', type: 'tool_request', content, result: null, name: toolName, params: toolParams, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId })

				// Special case: render_form and create_quiz never execute - they stay in tool_request state so the UI can display the interactive content
				if (toolName === 'render_form' || toolName === 'create_quiz') {
					return { awaitingUserApproval: true }
				}

				if (!autoApprove) {
					return { awaitingUserApproval: true }
				}
			}
		}
		else {
			toolParams = opts.validatedParams
		}

		// Use predictive progress message
		const progressMessage = this.getPredictiveProgressMessage(toolName, toolParams);
		const runningTool = { role: 'tool', type: 'running_now', name: toolName, params: toolParams, content: progressMessage, result: null, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId } as const
		this._updateToolMessage(threadId, runningTool, !!parallelMode)
		this._emitAgentEvent({ type: 'tool_execution_started', threadId, toolName })

		let interrupted = false
		let resolveInterruptor: (r: () => void) => void = () => { }
		const interruptorPromise = new Promise<() => void>(res => { resolveInterruptor = res })
		// Cancellation source for this tool call. Cancelled when the user aborts
		// the thread (abortRunning -> interrupt()), so in-flight search/context
		// tool calls that honour a CancellationToken stop early instead of running
		// to completion and landing their results in context.
		const cancellationTokenSource = new CancellationTokenSource()
		// Hoisted out of the try block below so the PostToolUse fire (which runs
		// after the try/catch, once toolResultStr is finalized) can read it.
		let preHookAdditionalContext: string | undefined
		try {

			// In parallel mode, skip stream state updates since multiple tools share the same thread state.
			// Only set stream state for sequential tool execution.
			if (!parallelMode) {
				this._setStreamState(threadId, { isRunning: 'tool', interrupt: interruptorPromise, toolInfo: { toolName, toolParams, id: toolId, content: progressMessage, rawParams: opts.unvalidatedToolParams, mcpServerName } })
			}

			// Fire PreToolUse hook. A `block` decision prevents the tool call (the
			// reason is fed back to the model as a tool error); `updatedInput` rewrites
			// the tool params. `ask` is treated as allow for v1 (the approval flow above
			// already handles user-consent gating). Fires after approval so it only
			// runs on the actual execution pass, not the approval-request pass.
			{
				const preHook = await this._hookService.firePreToolUse(threadId, toolName, toolParams as Record<string, unknown>)
				if (preHook.decision === 'block') {
					const blockReason = preHook.reason || `Tool ${toolName} was blocked by a PreToolUse hook.`
					this._updateToolMessage(threadId, { role: 'tool', type: 'tool_error', params: toolParams, result: blockReason, name: toolName, content: blockReason, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId }, !!parallelMode)
					this._emitAgentEvent({ type: 'tool_execution_finished', threadId, toolName, isError: true })
					return {}
				}
				if (preHook.updatedInput && typeof preHook.updatedInput === 'object') {
					toolParams = { ...(toolParams as object), ...preHook.updatedInput } as typeof toolParams
				}
				if (preHook.additionalContext) preHookAdditionalContext = preHook.additionalContext
			}

			// TOOL-RESULT CACHE — read-only deterministic tools (read_file, search_*, etc.)
			// are memoized per thread so repeated identical calls skip re-execution. Any
			// non-read-only tool (write/terminal/external) invalidates the whole thread
			// cache, since it may have changed what the cached reads would return. Runs
			// after PreToolUse so a param rewrite changes the lookup key.
			if (isBuiltInTool && READ_ONLY_TOOLS.has(toolName)) {
				if (!this.toolResultCache[threadId]) this.toolResultCache[threadId] = new Map()
				const cached = this.toolResultCache[threadId]!.get(this._toolCacheKey(toolName, toolParams))
				if (cached) {
					this._updateToolMessage(threadId, { role: 'tool', type: 'success', params: toolParams, result: cached.result, name: toolName, content: cached.resultStr, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId }, !!parallelMode)
					if (!this.toolCallHistory[threadId]) this.toolCallHistory[threadId] = []
					this.toolCallHistory[threadId].push({ name: toolName, params: toolParams, result: this._truncateToolResult(cached.resultStr), type: 'success', _paramsKey: this._getToolParamsKey(toolName, toolParams) })
					if (this.toolCallHistory[threadId].length > MAX_TOOL_CALL_HISTORY_PER_THREAD) this.toolCallHistory[threadId] = this.toolCallHistory[threadId].slice(-MAX_TOOL_CALL_HISTORY_PER_THREAD)
					this._emitAgentEvent({ type: 'tool_execution_finished', threadId, toolName, isError: false })
					return {}
				}
			}
			else {
				// A mutating or external tool is about to run — drop cached reads.
				delete this.toolResultCache[threadId]
			}

			if (isBuiltInTool) {
				const { result, interruptTool } = await this._toolsService.callTool[toolName](toolParams as any, {
					threadId,
					cancellationToken: cancellationTokenSource.token,
					onData: (data) => {
						// Stream partial results to the UI for immersion
						const currentStreamState = this.streamState[threadId];
						if (currentStreamState?.isRunning === 'tool') {
							// Update the content with the latest data (keep it brief)
							const truncatedData = data.length > 500 ? data.slice(-500) : data;
							this._setStreamState(threadId, {
								...currentStreamState,
								toolInfo: {
									...currentStreamState.toolInfo,
									content: truncatedData
								}
							});
						}
					}
				})
				const interruptor = () => { interrupted = true; interruptTool?.(); cancellationTokenSource.cancel() }
				resolveInterruptor(interruptor)

				toolResult = await result
			}
			else {
				// Check if this is a Composio tool
				if (mcpServerName === 'composio_tool_router') {
					// Composio Tool Router tools are handled via the Composio service
					resolveInterruptor(() => { })

					const sessionId = this._composioService.getSessionId()
					if (!sessionId) {
						throw new Error('Composio session not initialized. Please ensure your Composio API key is configured.')
					}

					const response = await this._composioService.executeToolViaSession(
						sessionId,
						this._composioService.getComposioSlug(toolName),
						toolParams as Record<string, unknown>
					)

					if (!response.successful) {
						throw new Error(response.error || 'Composio tool execution failed')
					}

					toolResult = response.data
				} else if (mcpServerName === 'acp_agent_router') {
					// ACP Agent tools - communicate with other agents
					resolveInterruptor(() => { })

					const acpTool = this._acpService.getACPAgents()?.find(t => t.name === toolName)
					if (!acpTool) {
						throw new Error(`ACP agent "${toolName}" not found`)
					}
					if (!acpTool.acpServerName || !acpTool.acpAgentName) {
						throw new Error(`ACP tool "${toolName}" is missing server or agent name`)
					}

					const input = (toolParams as { input: string }).input
					toolResult = (await this._acpService.callACPAgent({
						serverName: acpTool.acpServerName,
						agentName: acpTool.acpAgentName,
						input,
					})).result
				} else {
					// MCP tools
					const mcpTools = this._mcpService.getMCPTools()
					const mcpTool = mcpTools?.find(t => t.name === toolName)
					if (!mcpTool) { throw new Error(`MCP tool ${toolName} not found`) }
					if (!mcpTool.mcpServerName) { throw new Error(`MCP tool ${toolName} has no server name`) }

					resolveInterruptor(() => { })

					toolResult = (await this._mcpService.callMCPTool({
						serverName: mcpTool.mcpServerName,
						toolName: toolName,
						params: toolParams as Record<string, unknown>
					})).result
				}
			}

			if (interrupted) { return { interrupted: true } } // the tool result is added where we interrupt, not here
		}
		catch (error) {
			resolveInterruptor(() => { }) // resolve for the sake of it
			if (interrupted) { return { interrupted: true } } // the tool result is added where we interrupt, not here

			const errorMessage = getErrorMessage(error)
			this._updateToolMessage(threadId, { role: 'tool', type: 'tool_error', params: toolParams, result: errorMessage, name: toolName, content: errorMessage, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId }, !!parallelMode)
			this._emitAgentEvent({ type: 'tool_execution_finished', threadId, toolName, isError: true })
			return {}
		}
		finally {
			cancellationTokenSource.dispose()
		}
		// 4. stringify the result to give to the LLM
		try {
			if (isBuiltInTool) {
				toolResultStr = this._toolsService.stringOfResult[toolName](toolParams as any, toolResult as any)
			}
			// For Composio tools, handle result with TOON encoding support
			else if (mcpServerName === 'composio_tool_router') {
				// Handle null/undefined results
				if (toolResult === null || toolResult === undefined) {
					toolResultStr = 'Tool executed successfully with no output.'
				} else if (typeof toolResult === 'string') {
					toolResultStr = toolResult
				} else {
					// Try TOON encoding for structured results if enabled
					const enableToon = this._settingsService.state.globalSettings.enableToolResultTOON
					if (enableToon && this._toonService.shouldUseToon(toolResult)) {
						try {
							const toonEncoded = this._toonService.encode(toolResult)
							const jsonFallback = JSON.stringify(toolResult, null, 2)
							// Only use TOON if it saves space
							if (toonEncoded.length < jsonFallback.length * 0.9) {
								toolResultStr = `[TOON]\n${toonEncoded}`
							} else {
								toolResultStr = jsonFallback
							}
						} catch {
							toolResultStr = JSON.stringify(toolResult, null, 2)
						}
					} else {
						toolResultStr = JSON.stringify(toolResult, null, 2)
					}
				}
			}
			// For ACP agent tools, handle the response
			else if (mcpServerName === 'acp_agent_router') {
				const acpResult = toolResult as ACPRunAgentResponse
				toolResultStr = this._acpService.stringifyResult(acpResult)
			}
			// For MCP tools, handle the result based on its type
			else {
				toolResultStr = this._mcpService.stringifyResult(toolResult as RawMCPToolCall)
			}

			// LOOP DETECTION HINT: If we are repeating a failing call, add a hint for the agent
			const history = this.toolCallHistory[threadId] || [];
			// MEMORY OPTIMIZATION: Use fast key comparison instead of JSON.stringify on entire history
			const currentParamsKey = this._getToolParamsKey(toolName, toolParams);
			// Only check last 10 calls for performance
			const recentHistory = history.slice(-10);
			const isRepeat = recentHistory.some(h => {
				if (h.name !== toolName || h.type === 'success') return false;
				const historyParamsKey = h._paramsKey || this._getToolParamsKey(h.name, h.params);
				return historyParamsKey === currentParamsKey;
			});
			if (isRepeat) {
				toolResultStr += "\n\nNOTE: I've noticed you've tried this exact call before with a similar result. Please consider if you need to change your parameters, try a different tool, or ask the user for more information if you are stuck.";
			}
		} catch (error) {
			const errorMessage = this.toolErrMsgs.errWhenStringifying(error)
			const fullErrorStr = `${errorMessage}\n\nNOTE: If you've tried this before, consider a different approach.`;
			this._updateToolMessage(threadId, { role: 'tool', type: 'tool_error', params: toolParams, result: errorMessage, name: toolName, content: fullErrorStr, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId }, !!parallelMode)

			// Update history
			if (!this.toolCallHistory[threadId]) this.toolCallHistory[threadId] = [];
			this.toolCallHistory[threadId].push({ name: toolName, params: toolParams, result: this._truncateToolResult(errorMessage), type: 'error', _paramsKey: this._getToolParamsKey(toolName, toolParams) });
			// MEMORY OPTIMIZATION: Prune history if it exceeds max limit
			if (this.toolCallHistory[threadId].length > MAX_TOOL_CALL_HISTORY_PER_THREAD) {
				this.toolCallHistory[threadId] = this.toolCallHistory[threadId].slice(-MAX_TOOL_CALL_HISTORY_PER_THREAD);
			}

			// Auto-update task status when tools fail
			this._updateTaskStatusFromToolExecution(threadId, toolName, 'error')

			return {}
		}

		// Fire PostToolUse hook. `updatedToolOutput` replaces the result string the
		// model sees; `additionalContext` (from pre or post hooks) is appended to it.
		// Non-blocking: a hook error is logged and the original result is kept.
		try {
			const postHook = await this._hookService.firePostToolUse(threadId, toolName, toolParams as Record<string, unknown>, toolResultStr)
			if (postHook.updatedToolOutput) toolResultStr = postHook.updatedToolOutput
			if (postHook.additionalContext) toolResultStr = `${toolResultStr}\n\n${postHook.additionalContext}`
		} catch (err) {
			voidDevWarn('[hooks] PostToolUse fire threw (non-blocking):', err)
		}
		if (preHookAdditionalContext) toolResultStr = `${toolResultStr}\n\n${preHookAdditionalContext}`

		// 5. add to history and keep going
		this._updateToolMessage(threadId, { role: 'tool', type: 'success', params: toolParams, result: toolResult, name: toolName, content: toolResultStr, id: toolId, rawParams: opts.unvalidatedToolParams, mcpServerName, thought_signature: opts.thought_signature, parallelBatchId }, !!parallelMode)

		// TOOL-RESULT CACHE — store the post-hook result for read-only deterministic
		// tools so identical repeat calls skip execution (see lookup before execute).
		if (isBuiltInTool && READ_ONLY_TOOLS.has(toolName)) {
			if (!this.toolResultCache[threadId]) this.toolResultCache[threadId] = new Map()
			this.toolResultCache[threadId]!.set(this._toolCacheKey(toolName, toolParams), { resultStr: toolResultStr, result: toolResult })
		}

		// SIDE EFFECT: if it's load_skill, update the thread's loadedSkills
		if (toolName === 'load_skill') {
			const loadSkillResult = toolResult as { success: boolean; skill_name: string; instructions: string };
			if (loadSkillResult.success) {
				this.loadSkill(threadId, loadSkillResult.skill_name, loadSkillResult.instructions);
			}
		}

		// Update history
		if (!this.toolCallHistory[threadId]) this.toolCallHistory[threadId] = [];

		// MEMORY OPTIMIZATION: Truncate large tool results in history to prevent excessive memory usage
		const resultToStore = this._truncateToolResult(toolResult);

		this.toolCallHistory[threadId].push({ name: toolName, params: toolParams, result: resultToStore, type: 'success', _paramsKey: this._getToolParamsKey(toolName, toolParams) });
		// MEMORY OPTIMIZATION: Prune history if it exceeds max limit
		if (this.toolCallHistory[threadId].length > MAX_TOOL_CALL_HISTORY_PER_THREAD) {
			this.toolCallHistory[threadId] = this.toolCallHistory[threadId].slice(-MAX_TOOL_CALL_HISTORY_PER_THREAD);
		}

		// Auto-update task status when tools complete successfully
		this._updateTaskStatusFromToolExecution(threadId, toolName, 'success')
		this._emitAgentEvent({ type: 'tool_execution_finished', threadId, toolName, isError: false })

		return {}
	};




	private async _runChatAgent({
		threadId,
		modelSelection,
		modelSelectionOptions,
		callThisToolFirst,
		orchestrationResult,
	}: {
		threadId: string,
		modelSelection: ModelSelection | null,
		modelSelectionOptions: ModelSelectionOptions | undefined,
		callThisToolFirst?: ToolMessage<ToolName> & { type: 'tool_request' }
		orchestrationResult?: {
			suggestions: Array<{
				toolName: string;
				toolParams?: Record<string, any>;
				reasoning: string;
				confidence: 'high' | 'medium' | 'low';
			}>;
			reasoning: string;
			summary: string;
		};
	}) {


		let interruptedWhenIdle = false
		const idleInterruptor = Promise.resolve(() => { interruptedWhenIdle = true })
		// _runToolCall does not need setStreamState({idle}) before it, but it needs it after it. (handles its own setStreamState)

		// above just defines helpers, below starts the actual function
		const { chatMode } = this._settingsService.state.globalSettings // should not change as we loop even if user changes it, so it goes here
		const { overridesOfModel } = this._settingsService.state
		this._emitAgentEvent({ type: 'agent_run_started', threadId, chatMode })

		let nMessagesSent = 0
		let nPokesThisLoop = 0
		let nStopHookPokes = 0
		let shouldSendAnotherMessage = true
		let isRunningWhenEnd: IsRunningType = undefined

		// before enter loop, call tool
		if (callThisToolFirst) {
			const { interrupted } = await this._runToolCall(threadId, callThisToolFirst.name, callThisToolFirst.id, callThisToolFirst.mcpServerName, { preapproved: true, unvalidatedToolParams: callThisToolFirst.rawParams, validatedParams: callThisToolFirst.params, thought_signature: callThisToolFirst.thought_signature })
			if (interrupted) {
				this._setStreamState(threadId, undefined)
				this._addUserCheckpoint({ threadId })

			}
		}
		this._setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed' })  // just decorative, for clarity


		let lastYieldTime = Date.now()

		// tool use loop
		while (shouldSendAnotherMessage) {
			// PERFORMANCE: Yield to event loop if we've spent more than 16ms to prevent UI freezing
			if (Date.now() - lastYieldTime > 16) {
				await new Promise(resolve => setTimeout(resolve, 0));
				lastYieldTime = Date.now();
			}

			// false by default each iteration
			shouldSendAnotherMessage = false
			isRunningWhenEnd = undefined
			nMessagesSent += 1
			this._emitAgentEvent({ type: 'agent_turn_started', threadId, turnNumber: nMessagesSent })

			// Safety check: prevent infinite loops in agent mode
			const maxAgentIterations = this._settingsService.state.globalSettings.maxAgentIterations || 50
			if (nMessagesSent > maxAgentIterations) {
				voidDevWarn(`[chatThreadService] Agent mode exceeded maximum iterations (${maxAgentIterations}), stopping loop`)
				this._emitAgentEvent({ type: 'agent_run_max_iterations', threadId, maxIterations: maxAgentIterations })
				this._setStreamState(threadId, {
					isRunning: undefined,
					error: {
						message: `Agent exceeded maximum iterations (${maxAgentIterations}). The task may be too complex or the AI may be stuck in a loop.`,
						fullError: null
					}
				})
				break
			}

			this._setStreamState(threadId, { isRunning: 'idle', interrupt: idleInterruptor })

			const chatMessages = this.state.allThreads[threadId]?.messages ?? []
			const loadedSkills = this.state.allThreads[threadId]?.state.loadedSkills
			voidDevLog(`[_runChatAgent] threadId: ${threadId}, messages count: ${chatMessages.length}`);
			if (chatMessages.length > 0) {
				const lastMsg = chatMessages[chatMessages.length - 1];
				voidDevLog(`[_runChatAgent] Last message role: ${lastMsg.role}`);
				if (lastMsg.role === 'user') {
					voidDevLog(`[_runChatAgent] Last user message content length: ${lastMsg.content?.length || 0}`);
				}
			}
			let { messages, separateSystemMessage, tokenUsage, compressionStats } = await this._convertToLLMMessagesService.prepareLLMChatMessages({
				chatMessages,
				modelSelection,
				chatMode,
				loadedSkills,
				orchestrationResult,
				compaction: this.state.allThreads[threadId]?.state.compaction,
				usageAnchor: this.usageAnchorByThreadId.get(threadId),
			})
			// Snapshot of how many thread messages this prepare covered. Paired with
			// the real usage returned by the LLM, it forms the usage anchor that lets
			// later prepares count only the trailing delta instead of the whole thread.
			const threadMessagesAtPrepare = chatMessages.length

			// Fire PreCompact hook when a compaction actually ran on this turn, so
			// hooks can archive the full pre-compaction transcript. Fired after the
			// call (the compaction is internal to convertToLLMMessagesService); the
			// `chatMessages` snapshot above is what the hook receives. Non-blocking.
			if (compressionStats) {
				try {
					await this._hookService.firePreCompact(threadId, chatMessages)
				} catch (err) {
					voidDevWarn('[hooks] PreCompact fire threw (non-blocking):', err)
				}
			}

			// Notify UI when compression happened
			if (compressionStats) {
				triggerCompressionNotification(compressionStats, threadId);
			}

			// Update stream state with token usage
			this._setStreamState(threadId, { isRunning: 'idle', interrupt: idleInterruptor, tokenUsage })

			if (interruptedWhenIdle) {
				this._setStreamState(threadId, undefined)
				return
			}

			// Initialize ReAct parser for this iteration
			const reactParser = new StreamingXMLParser();
			let currentReActPhase: ReActPhase | null = null;
			let lastParsedLength = 0;

			let lastUpdateTime = 0;
			const UI_UPDATE_THROTTLE_MS = 50; // ~20 FPS - smoother for streaming, still CPU-friendly

			let shouldRetryLLM = true
			let nAttempts = 0
			while (shouldRetryLLM) {
				shouldRetryLLM = false
				nAttempts += 1

				type ResTypes =
					| { type: 'llmDone', toolCalls?: RawToolCallObj[], info: { fullText: string, fullReasoning: string, anthropicReasoning: AnthropicReasoning[] | null }, usage?: { promptTokens: number; completionTokens: number; cachedTokens?: number; }, stopReason?: string }
					| { type: 'llmError', error?: { message: string; fullError: Error | null; } }
					| { type: 'llmAborted' }

				let resMessageIsDonePromise: (res: ResTypes) => void // resolves when user approves this tool use (or if tool doesn't require approval)
				const messageIsDonePromise = new Promise<ResTypes>((res, rej) => { resMessageIsDonePromise = res })

				// Repetition detection: track last chunks to detect looping
				let lastChunks: string[] = [];
				const MAX_CHUNKS_TO_TRACK = 10;
				const REPETITION_THRESHOLD = 5; // If same chunk appears 5 times, it's looping
				
				// Out-of-order protection: enforce monotonic growth of raw text
				let maxRawLength = 0;

				const llmCancelToken = this._llmMessageService.sendLLMMessage({
					messagesType: 'chatMessages',
					chatMode,
					messages: messages,
					modelSelection,
					modelSelectionOptions,
					overridesOfModel,
					logging: { loggingName: `Chat - ${chatMode}`, loggingExtras: { threadId, nMessagesSent, chatMode } },
					separateSystemMessage: separateSystemMessage,
					onText: (params) => {
						let { fullText, fullReasoning, textDelta, reasoningDelta, toolCalls, _rawTextBeforeStripping } = params;
						
						// Enforce monotonic updates for raw text to prevent out-of-order rendering
						// We use raw text because display text can shrink when XML tags are stripped
						const currentRawLen = (_rawTextBeforeStripping || fullText).length;
						
						// If we receive a shorter text than seen before, it's likely an out-of-order packet
						// EXCEPT if it's a retry (length drops significantly to near zero)
						if (currentRawLen < maxRawLength) {
							// Allow reset if new length is very small (start of new stream)
							if (currentRawLen < 100 && maxRawLength > 200) {
								maxRawLength = currentRawLen; // Reset detected
							} else {
								// Ignore out-of-order update
								return;
							}
						} else {
							maxRawLength = currentRawLen;
						}

						// Backward compatibility for Main process running old code
						const legacyToolCall = (params as { toolCall?: RawToolCallObj }).toolCall;
						if (!toolCalls && legacyToolCall) {
							toolCalls = [legacyToolCall];
						}

						let parsed: { displayText: string, reasoningText: string };

						// PERFORMANCE: Use deltas if available to avoid O(N^2) processing
						// Actually, we must ALWAYS partition to handle <think> tags that might be in the stream
						parsed = partitionReasoningContent(fullText, fullReasoning)

						// If parsed content is empty and we have raw text, try to partition the raw text
						if (!parsed.displayText && !parsed.reasoningText && _rawTextBeforeStripping) {
							parsed = partitionReasoningContent(_rawTextBeforeStripping, fullReasoning);
						}

						// Parse ReAct phases for enhanced UI detection
						const textToParse = _rawTextBeforeStripping || fullText;
						const newChunk = textToParse.slice(lastParsedLength);
						lastParsedLength = textToParse.length;

						// Parse ReAct phases and XML tool calls together
						const reactResult = reactParser.parseReAct(newChunk);
						if (reactResult) {
							currentReActPhase = reactResult.phase;
						}

						                        // Detect repetition
						                        const hasXMLToolCallInProgress = _rawTextBeforeStripping?.includes('<function_calls>');
						                        const hasNativeToolCall = !!toolCalls && toolCalls.length > 0;
						                        
						                        if (!hasNativeToolCall && !hasXMLToolCallInProgress) {
						                            // Combine display text and reasoning for repetition detection
						                            // This prevents false positives when only reasoning is streaming
						                            const combinedText = (parsed.displayText + " " + parsed.reasoningText).trim();
						                            const recentText = combinedText.slice(-50);
						                            
						                            if (recentText.length > 10) {
						                                // Only add if the text has actually changed to avoid false positives 
						                                // from redundant onText calls (e.g. from provider heartbeats)
						                                if (lastChunks.length === 0 || lastChunks[lastChunks.length - 1] !== recentText) {
						                                    lastChunks.push(recentText);
						                                    if (lastChunks.length > MAX_CHUNKS_TO_TRACK) {
						                                        lastChunks.shift();
						                                    }
						                                }
						
						                                const repetitionCount = lastChunks.filter(chunk => chunk === recentText).length;
						                                if (repetitionCount >= REPETITION_THRESHOLD) {
						                                    voidDevWarn(`[chatThreadService] Text repetition detected. Count: ${repetitionCount}, Text: "${recentText.substring(0, 100)}..."`);
						                                    voidDevWarn(`[chatThreadService] Repetition threshold reached (${REPETITION_THRESHOLD}), aborting LLM...`);
						                                    if (llmCancelToken) {
						                                        this._llmMessageService.abort(llmCancelToken);
						                                    }
						                                    return;
						                                }
						                            }
						                        } else {
						                            // Reset tracker when tools are detected
						                            lastChunks = [];
						                        }
						// Use tool calls from ReAct parser if available, otherwise use native tool calls
						let parsedToolCalls = toolCalls;
						if (!parsedToolCalls && reactResult?.toolCalls) {
							parsedToolCalls = reactResult.toolCalls;
						}

						// Throttle UI updates
						const now = Date.now();
						const hasNewNativeToolCall = !!toolCalls && (toolCalls.length !== (this.streamState[threadId]?.llmInfo?.toolCallsSoFar?.length ?? 0));

						// MEMORY OPTIMIZATION: Only stringify compare if lengths match (avoid expensive stringify on every char)
						let hasUpdatedXMLToolCall = false;
						if (reactResult?.toolCalls) {
							const prevToolCalls = this.streamState[threadId]?.llmInfo?.toolCallsSoFar;
							// Quick length check first
							if (!prevToolCalls || reactResult.toolCalls.length !== prevToolCalls.length) {
								hasUpdatedXMLToolCall = true;
							} else if (reactResult.isComplete) {
								// Only do expensive stringify comparison when complete, not on every char
								hasUpdatedXMLToolCall = JSON.stringify(reactResult.toolCalls) !== JSON.stringify(prevToolCalls);
							}
						}
						
						const isCriticalUpdate = hasNewNativeToolCall || hasUpdatedXMLToolCall || reactResult?.isComplete;
						
						if (now - lastUpdateTime < UI_UPDATE_THROTTLE_MS && !isCriticalUpdate) {
							return;
						}
						lastUpdateTime = now;

						this._setStreamState(threadId, {
							isRunning: 'LLM',
							llmInfo: {
								displayContentSoFar: parsed.displayText,
								reasoningSoFar: parsed.reasoningText,
								toolCallsSoFar: parsedToolCalls ?? null,
								_rawTextBeforeStripping,
								reactPhase: currentReActPhase,
								textDelta,
								reasoningDelta,
							},
							interrupt: Promise.resolve(() => { if (llmCancelToken) this._llmMessageService.abort(llmCancelToken) }),
							tokenUsage,
						})
					},
					onFinalMessage: async (params) => {
						let { fullText, fullReasoning, toolCalls, anthropicReasoning, usage } = params;
						// Backward compatibility for Main process running old code
						const legacyToolCall = (params as { toolCall?: RawToolCallObj }).toolCall;
						if (!toolCalls && legacyToolCall) {
							toolCalls = [legacyToolCall];
						}

						voidDevLog(`[chatThreadService] onFinalMessage received - fullReasoning length: ${fullReasoning?.length ?? 0}, toolCalls: ${toolCalls?.length ?? 0}`)
						if (usage) {
							voidDevLog(`[chatThreadService] Token usage received:`, usage);
							// Record the usage anchor: this call's measured promptTokens is
							// ground truth for the `threadMessagesAtPrepare` messages that were
							// on the wire, so later prepares only estimate the trailing delta.
							if (usage.promptTokens > 0 && modelSelection) {
								this.usageAnchorByThreadId.set(threadId, {
									promptTokens: usage.promptTokens,
									coveredThreadMessageCount: threadMessagesAtPrepare,
									compactedChatMessageCountAtAnchor: this.state.allThreads[threadId]?.state.compaction?.compactedChatMessageCount ?? 0,
									providerName: modelSelection.providerName,
									modelName: modelSelection.modelName,
								})
							}
							// Update token ratio for adaptive counting
							if (tokenUsage?.used && usage.promptTokens) {
								const { providerName, modelName } = modelSelection!;
								const fullModelName = `${providerName}:${modelName}`;
								// tokenUsage.used is our estimate, usage.promptTokens is actual
								// We pass both so the service can calculate and update the ratio
								// We access the service via the public method
								this._convertToLLMMessagesService.updateTokenRatio(fullModelName, tokenUsage.used, usage.promptTokens);
							}
						}
						const parsed = partitionReasoningContent(fullText, fullReasoning)
						voidDevLog(`[chatThreadService] After partitioning - reasoningText length: ${parsed.reasoningText?.length ?? 0}`)
						resMessageIsDonePromise({ type: 'llmDone', toolCalls, info: { fullText: parsed.displayText, fullReasoning: parsed.reasoningText, anthropicReasoning }, usage, stopReason: params.stopReason }) // resolve with tool calls
					},
					onError: async (error) => {
						resMessageIsDonePromise({ type: 'llmError', error: error })
					},
					onAbort: () => {
						// stop the loop to free up the promise, but don't modify state (already handled by whatever stopped it)
						resMessageIsDonePromise({ type: 'llmAborted' })
						this._metricsService.capture('Agent Loop Done (Aborted)', { nMessagesSent, chatMode })
					},
				})

				// mark as streaming
				if (!llmCancelToken) {
					this._setStreamState(threadId, { isRunning: undefined, error: { message: 'There was an unexpected error when sending your chat message.', fullError: null } })
					break
				}

				this._setStreamState(threadId, { isRunning: 'LLM', llmInfo: { displayContentSoFar: '', reasoningSoFar: '', toolCallsSoFar: null, reactPhase: null }, interrupt: Promise.resolve(() => this._llmMessageService.abort(llmCancelToken)) })
				this._emitAgentEvent({ type: 'llm_request_started', threadId, messageCount: threadMessagesAtPrepare })
				const llmRes = await messageIsDonePromise // wait for message to complete

				// if something else started running in the meantime
				if (this.streamState[threadId]?.isRunning !== 'LLM') {
					// voidDevLog('Chat thread interrupted by a newer chat thread', this.streamState[threadId]?.isRunning)
					return
				}

				// llm res aborted
				if (llmRes.type === 'llmAborted') {
					this._emitAgentEvent({ type: 'llm_aborted', threadId })
					this._setStreamState(threadId, undefined)
					this._emitAgentEvent({ type: 'agent_run_finished', threadId, outcome: 'aborted' })
					return
				}
				// llm res error
				else if (llmRes.type === 'llmError') {
					const errorMsg = llmRes.error?.message || '';
					const isContextError = CONTEXT_LENGTH_ERROR_RE.test(errorMsg);

					// Handle context length errors specifically by adjusting token estimation
					if (isContextError && nAttempts < CHAT_RETRIES) {
						voidDevWarn(`[chatThreadService] Context length error detected: ${errorMsg}`);
						const { providerName, modelName } = modelSelection!;
						const fullModelName = `${providerName}:${modelName}`;
						
						// Force a more conservative ratio
						// Since we can't get actual usage on error, we just blindly increase the multiplier
						// This tells the token service "whatever you thought the count was, it's actually 1.5x higher"
						// We pass dummy values (estimated=1000, actual=1500) to force a 1.5 ratio update
						this._convertToLLMMessagesService.updateTokenRatio(fullModelName, 1000, 1500);
						voidDevLog(`[chatThreadService] Bumped token ratio for ${fullModelName} due to context error`);
						
						shouldRetryLLM = true;
						this._setStreamState(threadId, {
							isRunning: undefined,
							error: { message: `Context limit hit, compressing and retrying... (attempt ${nAttempts}/${CHAT_RETRIES})`, fullError: null }
						});
						
						// Re-prepare messages with new ratio (this will trigger compression)
						const newPrep = await this._convertToLLMMessagesService.prepareLLMChatMessages({
							chatMessages,
							modelSelection,
							chatMode,
							compaction: this.state.allThreads[threadId]?.state.compaction,
							usageAnchor: this.usageAnchorByThreadId.get(threadId),
						});

						// Update messages and token usage for the retry
						messages = newPrep.messages;
						separateSystemMessage = newPrep.separateSystemMessage;
						tokenUsage = newPrep.tokenUsage;


						// Notify UI when compression happened on retry
						if (newPrep.compressionStats) {
							triggerCompressionNotification(newPrep.compressionStats, threadId);
						}

						this._setStreamState(threadId, { isRunning: 'idle', interrupt: idleInterruptor, tokenUsage });
					}

					// error, should retry
					if (nAttempts < CHAT_RETRIES) {
						shouldRetryLLM = true
						voidDevLog(`[chatThreadService] LLM error, retrying (attempt ${nAttempts}/${CHAT_RETRIES})...`)
						// Show retry message briefly
						this._setStreamState(threadId, {
							isRunning: undefined,
							error: { message: `Retrying... (attempt ${nAttempts}/${CHAT_RETRIES})`, fullError: null }
						})
						await timeout(RETRY_DELAY_BASE * Math.pow(2, nAttempts - 1))
						if (interruptedWhenIdle) {
							this._setStreamState(threadId, undefined)
							return
						}
						else {
							// Clear error before retry
							this._setStreamState(threadId, { isRunning: 'idle', interrupt: idleInterruptor })
							
							// Note: messages have already been re-prepared if it was a context error
							
							continue // retry
						}
					}
					// error, but too many attempts
					else {
						const { error } = llmRes
						const { displayContentSoFar, reasoningSoFar, toolCallsSoFar } = this.streamState[threadId].llmInfo
						this._addMessageToThread(threadId, { role: 'assistant', displayContent: displayContentSoFar, reasoning: reasoningSoFar, anthropicReasoning: null })
						if (toolCallsSoFar) {
							for (const tc of toolCallsSoFar) {
								this._addMessageToThread(threadId, { role: 'interrupted_streaming_tool', name: tc.name, mcpServerName: this._computeMCPServerOfToolName(tc.name) })
							}
						}

						// Fire StopFailure hook (non-blocking: hooks can log/react but
						// cannot recover the run). Errors here are swallowed.
						try {
							const sfHook = await this._hookService.fireStopFailure(threadId, error?.message ?? 'LLM error after all retries')
							if (sfHook.continue === false) {
								voidDevWarn('[hooks] StopFailure hook requested continue, but the run has errored and cannot recover.')
							}
						} catch (err) {
							voidDevWarn('[hooks] StopFailure fire threw (non-blocking):', err)
						}

						this._setStreamState(threadId, { isRunning: undefined, error })
						this._emitAgentEvent({ type: 'llm_errored', threadId, errorMessage: error?.message ?? '' })
						this._emitAgentEvent({ type: 'agent_run_finished', threadId, outcome: 'error' })
						return
					}
				}

				// llm res success
				const { toolCalls, info, stopReason, usage: llmUsage } = llmRes
				this._emitAgentEvent({ type: 'llm_response_received', threadId, toolCallCount: toolCalls?.length ?? 0, stopReason })

									const responseLog = JSON.stringify({
										hasToolCalls: !!toolCalls && toolCalls.length > 0,
										toolCallsCount: toolCalls?.length ?? 0,
										fullText: info.fullText,
										reasoning: info.fullReasoning
									});
									voidDevLog(`[chatThreadService] LLM response:`, responseLog.length > 1000 ? responseLog.substring(0, 1000) + '...' : responseLog)
								// Check for empty response and treat as error for retry
				// Note: Tool calls with empty content are valid (especially for Ollama)
				// Also treat "(empty message)" placeholder as empty
				const textContent = info.fullText?.trim() || ''
				const isEmptyResponse = (textContent.length === 0 || textContent === '(empty message)') && (!toolCalls || toolCalls.length === 0) && !info.fullReasoning && (!info.anthropicReasoning || info.anthropicReasoning.length === 0)
				if (isEmptyResponse) {
					// In both modes, retry with delay if we haven't exhausted attempts
					if (nAttempts < CHAT_RETRIES) {
						shouldRetryLLM = true
						voidDevWarn(`[chatThreadService] LLM returned empty response, retrying (attempt ${nAttempts}/${CHAT_RETRIES})...`)
						
						// Show retry message briefly
						this._setStreamState(threadId, {
							isRunning: undefined,
							error: { message: `Empty response, retrying... (attempt ${nAttempts}/${CHAT_RETRIES})`, fullError: null }
						})
						
						await timeout(RETRY_DELAY_BASE * Math.pow(2, nAttempts - 1))
						if (interruptedWhenIdle) {
							this._setStreamState(threadId, undefined)
							return
						}
						else {
							// Clear error before retry
							this._setStreamState(threadId, { isRunning: 'idle', interrupt: idleInterruptor })
							continue // retry current turn
						}
					}
					// Empty response but too many attempts
					else {
						console.error(`[chatThreadService] LLM returned empty response after ${CHAT_RETRIES} attempts, giving up`)
						
						if (chatMode === 'code') {
							// In agent mode, instead of just breaking, try adding a "poke" message to break the cycle
							// Only do this once to avoid infinite poking
							const messages = this.state.allThreads[threadId]?.messages || []
							const lastPokeIdx = findLastIdx(messages, m => m.role === 'user' && m.content.includes('I received an empty response'))
							
							if (lastPokeIdx === -1 || messages.length - lastPokeIdx > 2) {
								voidDevLog('[chatThreadService] Agent mode: Adding poke message after empty responses')
								this._addMessageToThread(threadId, { 
									role: 'user', 
									content: 'I received an empty response from you. If you are stuck, please try a different approach or ask me for clarification. Otherwise, please continue with the task.',
									displayContent: 'Continuing after empty response...',
									selections: null,
									state: defaultMessageState
								})
								shouldSendAnotherMessage = true
								break // Exit retry loop to start a new turn with the poke message
							}
						}

						this._setStreamState(threadId, {
							isRunning: undefined,
							error: {
								message: `LLM returned empty response after ${CHAT_RETRIES} attempts. Please try again or check your model configuration.`,
								fullError: null
							}
						})
						break // Exit retry loop
					}
				}

				// Only add non-empty messages to thread
				if (!isEmptyResponse) {
					voidDevLog(`[chatThreadService] Adding assistant message with reasoning length: ${info.fullReasoning?.length ?? 0}`)
					this._addMessageToThread(threadId, { role: 'assistant', displayContent: info.fullText, reasoning: info.fullReasoning, anthropicReasoning: info.anthropicReasoning })
				}

				this._setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed', stopReason, cachedTokens: llmUsage?.cachedTokens }) // just decorative for clarity

				// call tool(s) if there are any
				if (toolCalls && toolCalls.length > 0) {
					const mcpTools = this._mcpService.getMCPTools()
					voidDevLog(`[chatThreadService] LLM called ${toolCalls.length} tool(s)`)

					let anyToolRan = false;

					// DYNAMIC PARALLEL SAFETY: Analyze tool calls for parallel execution safety.
					// This considers tool types, target resources, and potential conflicts.
					const { parallelSafe, sequential } = analyzeParallelToolSafety(toolCalls)

					// Additional check: auto-approval status affects parallel safety
					// Even read-only tools need auto-approval for parallel execution
					const finalParallelSafe: typeof toolCalls = []
					const finalSequential: typeof toolCalls = [...sequential]

					for (const toolCall of parallelSafe) {
						const isBuiltin = isABuiltinToolName(toolCall.name)
						const approvalType = isBuiltin ? approvalTypeOfBuiltinToolName[toolCall.name as BuiltinToolName] : 'MCP tools'
						const needsApproval = !!approvalType
						// Terminal tools use the layered command-allowlist policy; for the
						// grouping decision we read the command from the raw params (the
						// authoritative approval still happens in _runToolCall with the
						// validated params). Other types use the simple per-type boolean.
						const autoApprove = needsApproval
							? approvalType === 'terminal'
								? shouldAutoApproveTerminalTool(toolCall.name as ToolName, typeof toolCall.rawParams?.command === 'string' ? toolCall.rawParams.command : undefined, this._terminalAutoApproveSettings())
								: !!this._settingsService.state.globalSettings.autoApprove[approvalType]
							: true

						// Parallel execution requires auto-approval
						if (autoApprove) {
							finalParallelSafe.push(toolCall)
						} else {
							finalSequential.push(toolCall)
						}
					}

					// Run parallel-safe tools concurrently when there are 2+ read-only tools.
					// Uses parallelMode=true in _runToolCall which appends messages instead of
					// replacing the last one, preventing concurrent tools from overwriting each other.
					if (finalParallelSafe.length > 1) {
						const parallelBatchId = generateUuid();
						voidDevLog(`[chatThreadService] Running ${finalParallelSafe.length} read-only tools in parallel: ${finalParallelSafe.map(t => t.name).join(', ')} (batch: ${parallelBatchId})`)
						const parallelResults = await Promise.all(
							finalParallelSafe.map(toolCall => {
								const mcpServerName = this._computeMCPServerOfToolName(toolCall.name);
								return this._runToolCall(threadId, toolCall.name, toolCall.id, mcpServerName, { preapproved: false, unvalidatedToolParams: toolCall.rawParams, thought_signature: toolCall.thought_signature }, true, parallelBatchId);
							})
						)
						for (const result of parallelResults) {
							if (result.interrupted) {
								this._setStreamState(threadId, undefined)
								return
							}
							if (result.awaitingUserApproval) {
								isRunningWhenEnd = 'awaiting_user'
								shouldSendAnotherMessage = false
							} else {
								anyToolRan = true
							}
						}
					} else if (finalParallelSafe.length === 1) {
						// Single parallel-safe tool, run normally (no batch grouping needed)
						const toolCall = finalParallelSafe[0]
						voidDevLog(`[chatThreadService] LLM calling tool: ${toolCall.name}`)
						const mcpServerName = this._computeMCPServerOfToolName(toolCall.name);
						const { awaitingUserApproval, interrupted } = await this._runToolCall(threadId, toolCall.name, toolCall.id, mcpServerName, { preapproved: false, unvalidatedToolParams: toolCall.rawParams, thought_signature: toolCall.thought_signature })
						if (interrupted) {
							this._setStreamState(threadId, undefined)
							return
						}
						if (awaitingUserApproval) {
							isRunningWhenEnd = 'awaiting_user'
							shouldSendAnotherMessage = false
						} else {
							anyToolRan = true
						}
					}

					// Run sequential tools one at a time
					for (const toolCall of finalSequential) {
						voidDevLog(`[chatThreadService] LLM calling tool: ${toolCall.name}`)
						const paramsStr = JSON.stringify(toolCall.rawParams);
						voidDevLog(`[chatThreadService] Tool call params:`, paramsStr.length > 1000 ? paramsStr.substring(0, 1000) + '...' : paramsStr)
						const mcpTool = mcpTools?.find(t => t.name === toolCall.name)

						// Determine mcpServerName - check for Composio tools first
						let mcpServerName: string | undefined = mcpTool?.mcpServerName
						const isComposio = this._composioService.isComposioTool(toolCall.name);
						voidDevLog(`[chatThreadService] Tool "${toolCall.name}" - mcpTool found: ${!!mcpTool}, isComposio=${isComposio}`);
						if (isComposio) {
							mcpServerName = 'composio_tool_router'
						}

						const { awaitingUserApproval, interrupted } = await this._runToolCall(threadId, toolCall.name, toolCall.id, mcpServerName, { preapproved: false, unvalidatedToolParams: toolCall.rawParams, thought_signature: toolCall.thought_signature })
						if (interrupted) {
							this._setStreamState(threadId, undefined)
							return
						}

						if (awaitingUserApproval) {
							isRunningWhenEnd = 'awaiting_user';
							shouldSendAnotherMessage = false;
							break; // STOP here, wait for user
						} else {
							anyToolRan = true;
						}
					}

					if (!isRunningWhenEnd && anyToolRan) {
						shouldSendAnotherMessage = true;
					}

					this._setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed', cachedTokens: llmUsage?.cachedTokens }) // just decorative, for clarity
				}
				// Handle text-only responses (no tool call)
				// Following Claude Code / Continue pattern: If no tool call, task is complete.
				// The LLM knows when it needs to use tools - if it responds with just text, it's done.
				else if (!isEmptyResponse) {
					if (chatMode === 'code') {
						const thread = this.state.allThreads[threadId];
						const workflow = thread?.state.activeWorkflow;

						// NEW: If active workflow has pending tasks, continue loop
						const hasPendingTasks = workflow && workflow.status === 'active' &&
							workflow.tasks.some(t => t.status === 'pending' || t.status === 'in_progress');

						// max_tokens truncation → continue (bounded + context-guarded)
						const canonicalStop = normalizeStopReason(stopReason)
						if (canonicalStop === 'max_tokens' && nPokesThisLoop < 3) {
							// Context guard: if context is nearly full, re-poking would just truncate again.
							// Fall through to the natural-stop path so compression can run next turn.
							if (tokenUsage && tokenUsage.percentage > 90) {
								voidDevWarn(`[chatThreadService] max_tokens truncation but context at ${tokenUsage.percentage}% — not poking; letting compression run.`)
							} else {
								voidDevLog(`[chatThreadService] Agent mode: Response truncated (max_tokens), continuing...`)
								nPokesThisLoop += 1
								this._addMessageToThread(threadId, {
									role: 'user',
									content: 'You were cut off mid-response before finishing. Continue exactly where you left off — do not repeat what you already produced, just complete the remaining output.',
									displayContent: 'Continuing truncated response...',
									selections: null,
									state: defaultMessageState
								})
								shouldSendAnotherMessage = true
								break
							}
						}

						if (hasPendingTasks) {
							voidDevLog(`[chatThreadService] Active workflow has pending tasks, continuing...`);
							shouldSendAnotherMessage = true;
							break; // Break retry loop to start new turn
						}

						// Detect interrupted responses (dangling intent or unfinished XML)
						const isInterruptedXML = reactParser.isParsingIncomplete();
						const danglingIntent = detectDanglingAgenticIntent(info.fullText, info.fullReasoning);

						if ((isInterruptedXML || danglingIntent !== 'none') && nPokesThisLoop < 3) {
							nPokesThisLoop += 1;

							if (danglingIntent === 'silent') {
								voidDevLog(`[chatThreadService] Agent mode: Detected obvious 'About to Act' pattern, silently auto-continuing...`)
								// Silent auto-continue: just start another turn without adding a user message
								shouldSendAnotherMessage = true;
							} else {
								voidDevLog(`[chatThreadService] Agent mode: Detected interrupted response (XML incomplete: ${isInterruptedXML}, intent: ${danglingIntent}). Poking model...`)
								this._addMessageToThread(threadId, {
									role: 'user',
									content: 'Your last response seemed interrupted or you mentioned an action without calling the corresponding tool. Please continue and call the tool now. Do not repeat your thought process, just proceed with the tool call.',
									displayContent: 'Continuing interrupted response...',
									selections: null,
									state: defaultMessageState
								})
								shouldSendAnotherMessage = true;
							}
							break; // Break retry loop to start new turn
						}

						// If the response is very short after a tool call, it might be an accidental termination
						// (e.g. just saying "Done." or "Okay." without actually being finished)
						const isVeryShortResponse = textContent.length < 25;
						const lastMessageWasToolResult = chatMessages.length > 0 && chatMessages[chatMessages.length - 1].role === 'tool';

						if (isVeryShortResponse && lastMessageWasToolResult && nPokesThisLoop < 3) {
							voidDevLog(`[chatThreadService] Agent mode: Model returned short response (${textContent.length} chars) after tool call, silently auto-continuing...`)
							nPokesThisLoop += 1;
							shouldSendAnotherMessage = true;
							break;
						}

						// Handle models that output ONLY reasoning (thinking) without text or tool calls
						// These models (like Gemini 3 Pro with thinking) will reason then stop, expecting to continue
						const isOnlyReasoning = info.fullReasoning && info.fullReasoning.length > 10 && textContent.length === 0;
						if (isOnlyReasoning && nPokesThisLoop < 3) {
							voidDevLog(`[chatThreadService] Agent mode: Model returned reasoning only (${info.fullReasoning.length} chars) without text or tool call, silently auto-continuing...`)
							nPokesThisLoop += 1;
							shouldSendAnotherMessage = true;
							break;
						}

						// Natural stop: fire the Stop hook whenever the model gave a text-only
						// response and there are no pending workflow tasks — regardless of
						// workflow status (previously this only fired when the workflow was
						// completed/null, so planning/paused/failed states skipped /goal).
						if (!hasPendingTasks) {
							try {
								const stopHook = await this._hookService.fireStop(threadId)
								if (stopHook.continue === false) {
									if (nStopHookPokes < MAX_STOP_HOOK_POKES) {
										nStopHookPokes += 1
										const pokeReason = stopHook.reason || 'A Stop hook requested the agent to keep working.'
										voidDevLog(`[chatThreadService] Stop hook says continue=false (${nStopHookPokes}/${MAX_STOP_HOOK_POKES}): ${pokeReason}`)
										this._addMessageToThread(threadId, {
											role: 'user',
											content: pokeReason,
											displayContent: pokeReason,
											selections: null,
											state: defaultMessageState
										})
										shouldSendAnotherMessage = true
										break
									} else {
										voidDevWarn(`[chatThreadService] Stop hook keep-working request hit cap (${MAX_STOP_HOOK_POKES}); stopping to avoid runaway.`)
									}
								}
							} catch (err) {
								voidDevWarn('[hooks] Stop fire threw (non-blocking):', err)
							}
							voidDevLog(`[chatThreadService] Agent mode: Text-only response (no tool call) - task complete`)
							shouldSendAnotherMessage = false
							break
						}
					}
				} // end while (attempts)
			} // end while (send message)

			// if awaiting user approval, keep isRunning true, else end isRunning
			this._setStreamState(threadId, { isRunning: isRunningWhenEnd })

			// add checkpoint before the next user message
			if (!isRunningWhenEnd) this._addUserCheckpoint({ threadId })

			// capture number of messages sent
			this._metricsService.capture('Agent Loop Done', { nMessagesSent, chatMode })
			this._emitAgentEvent({ type: 'agent_run_finished', threadId, outcome: 'done' })

			// Process next queued message if any
			if (!isRunningWhenEnd) {
				const thread = this.state.allThreads[threadId];
				const workflow = thread?.state.activeWorkflow;
				const hasActiveWorkflow = workflow &&
					workflow.status === 'active' &&
					workflow.tasks.some(t => t.status === 'pending' || t.status === 'in_progress');

				if (hasActiveWorkflow && thread?.state.queueBehavior === 'wait_for_workflow') {
					voidDevLog('[chatThreadService] Workflow active, holding queued message');
					// Don't process - wait for workflow completion
				} else {
					await this._processNextQueuedMessage(threadId);
				}
			}
		}
	}


	private _addCheckpoint(threadId: string, checkpoint: CheckpointEntry) {
		this._addMessageToThread(threadId, checkpoint)
		// // update latest checkpoint idx to the one we just added
		// const newThread = this.state.allThreads[threadId]
		// if (!newThread) return // should never happen
		// const currCheckpointIdx = newThread.messages.length - 1
		// this._setThreadState(threadId, { currCheckpointIdx: currCheckpointIdx })
	}



	private _editMessageInThread(threadId: string, messageIdx: number, newMessage: ChatMessage,) {
		const { allThreads } = this.state
		const oldThread = allThreads[threadId]
		if (!oldThread) return // should never happen
		// update state and store it
		const newThreads = {
			...allThreads,
			[oldThread.id]: {
				...oldThread,
				lastModified: new Date().toISOString(),
				messages: [
					...oldThread.messages.slice(0, messageIdx),
					newMessage,
					...oldThread.messages.slice(messageIdx + 1, Infinity),
				],
			}
		}
		this._storeAllThreads(newThreads)
		this._setState({ allThreads: newThreads }) // the current thread just changed (it had a message added to it)
	}


	private _getCheckpointInfo = (checkpointMessage: ChatMessage & { role: 'checkpoint' }, fsPath: string, opts: { includeUserModifiedChanges: boolean }): { voidFileSnapshot: VoidFileSnapshot | null } | undefined => {
		// Try new diff-based format first
		const diffCheckpoint = checkpointMessage.diffBasedCheckpointsOfURI?.[fsPath];
		if (diffCheckpoint) {
			// For diff-based checkpoints, we need to reconstruct the full content
			// by walking back through the checkpoint chain
			const fullContent = this._reconstructFileContentFromDiffs(checkpointMessage, fsPath);
			if (fullContent !== null) {
				return {
					voidFileSnapshot: {
						snapshottedDiffAreaOfId: diffCheckpoint.snapshottedDiffAreaOfId,
						entireFileCode: fullContent
					}
				};
			}
			return undefined;
		}

		// Fall back to legacy format
		const voidFileSnapshot = checkpointMessage.voidFileSnapshotOfURI?.[fsPath] ?? null;
		if (!voidFileSnapshot) return undefined;

		if (!opts.includeUserModifiedChanges) {
			return { voidFileSnapshot };
		}

		const userModifiedSnapshot = checkpointMessage.userModifications?.voidFileSnapshotOfURI?.[fsPath];
		return { voidFileSnapshot: userModifiedSnapshot ?? voidFileSnapshot };
	}

	/**
	 * Reconstruct file content by walking back through diff-based checkpoints
	 * and applying diffs from the earliest full snapshot
	 */
	private _reconstructFileContentFromDiffs(checkpointMessage: ChatMessage & { role: 'checkpoint' }, fsPath: string): string | null {
		const diffCheckpoint = checkpointMessage.diffBasedCheckpointsOfURI?.[fsPath];
		if (!diffCheckpoint) return null;

		// If this is a full snapshot, just return the content directly
		if (diffCheckpoint.isFullSnapshot) {
			return diffCheckpoint.fileContentDiffs[0]?.newText || null;
		}

		// For diff-based checkpoints, we need to walk back to find the first full snapshot
		// This is a simplified version - in practice we might want to cache reconstructed contents
		const threadId = this.state.currentThreadId;
		const thread = this.state.allThreads[threadId];
		if (!thread) return null;

		// Find the checkpoint index
		const checkpointIdx = thread.messages.findIndex(m => m === checkpointMessage);
		if (checkpointIdx === -1) return null;

		// Walk back to find a full snapshot
		let currentContent: string | null = null;
		const diffsToApply: typeof diffCheckpoint.fileContentDiffs = [];

		for (let i = checkpointIdx; i >= 0; i--) {
			const message = thread.messages[i];
			if (message.role !== 'checkpoint') continue;

			const checkpoint = message.diffBasedCheckpointsOfURI?.[fsPath];
			if (!checkpoint) {
				// Check legacy format
				const legacySnapshot = message.voidFileSnapshotOfURI?.[fsPath];
				if (legacySnapshot) {
					currentContent = legacySnapshot.entireFileCode;
					break;
				}
				continue;
			}

			if (checkpoint.isFullSnapshot) {
				currentContent = checkpoint.fileContentDiffs[0]?.newText || '';
				break;
			}

			// Collect diffs to apply (in reverse order)
			diffsToApply.unshift(...checkpoint.fileContentDiffs);
		}

		// If we found a base content, apply all collected diffs
		if (currentContent !== null) {
			// Apply diffs in order
			for (const diff of diffsToApply) {
				currentContent = applyDiffBasedCheckpoint(currentContent, {
					...diffCheckpoint,
					fileContentDiffs: [diff],
					isFullSnapshot: false
				});
			}
			// Finally apply the target checkpoint's diffs
			for (const diff of diffCheckpoint.fileContentDiffs) {
				currentContent = applyDiffBasedCheckpoint(currentContent, {
					...diffCheckpoint,
					fileContentDiffs: [diff],
					isFullSnapshot: false
				});
			}
		}

		return currentContent;
	}

	private _computeNewCheckpointInfo({ threadId }: { threadId: string }) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		const lastCheckpointIdx = findLastIdx(thread.messages, (m) => m.role === 'checkpoint') ?? -1

		// MEMORY OPTIMIZATION: Store diffs instead of full snapshots
		const diffBasedCheckpointsOfURI: { [fsPath: string]: DiffBasedCheckpoint | undefined } = {}
		const previousCheckpointContents: { [fsPath: string]: string } = {}

		// Only process files that have actually changed to save compute
		for (const fsPath of thread.filesWithUserChanges) {
			const { model } = this._voidModelService.getModelFromFsPath(fsPath)
			if (!model) continue

			const newSnapshot = this._editCodeService.getVoidFileSnapshot(URI.file(fsPath))
			let previousSnapshot: VoidFileSnapshot | null = null

			// Find the last checkpoint for this specific file to compare
			if (lastCheckpointIdx !== -1) {
				const { lastIdxOfURI } = this._getCheckpointsBetween({ threadId, loIdx: 0, hiIdx: lastCheckpointIdx })
				const lastCheckpointIdxForFile = lastIdxOfURI[fsPath]

				if (lastCheckpointIdxForFile !== undefined) {
					const lastCheckpoint = thread.messages[lastCheckpointIdxForFile]
					if (lastCheckpoint.role === 'checkpoint') {
						// Get previous content from diff-based checkpoint
						const prevDiffCheckpoint = lastCheckpoint.diffBasedCheckpointsOfURI?.[fsPath];
						if (prevDiffCheckpoint && previousCheckpointContents[fsPath]) {
							// Reconstruct previous snapshot from diff
							const prevContent = previousCheckpointContents[fsPath];
							const prevFullContent = applyDiffBasedCheckpoint(prevContent, prevDiffCheckpoint);
							previousSnapshot = {
								snapshottedDiffAreaOfId: prevDiffCheckpoint.snapshottedDiffAreaOfId,
								entireFileCode: prevFullContent
							};
						} else {
							// Fall back to legacy format
							const res = this._getCheckpointInfo(lastCheckpoint, fsPath, { includeUserModifiedChanges: false })
							if (res?.voidFileSnapshot) {
								previousSnapshot = res.voidFileSnapshot
								previousCheckpointContents[fsPath] = res.voidFileSnapshot.entireFileCode;
							}
						}
					}
				}
			}

			// Create diff-based checkpoint
			const diffCheckpoint = createDiffBasedCheckpoint(previousSnapshot, newSnapshot)

			// Only store if there are actual changes or if it's the first checkpoint for this file
			if (diffCheckpoint.fileContentDiffs.length > 0 || !previousSnapshot) {
				diffBasedCheckpointsOfURI[fsPath] = diffCheckpoint
			}
		}

		return { diffBasedCheckpointsOfURI, previousCheckpointIdx: lastCheckpointIdx >= 0 ? lastCheckpointIdx : null }
	}


	private _addUserCheckpoint({ threadId }: { threadId: string }) {
		const { diffBasedCheckpointsOfURI, previousCheckpointIdx } = this._computeNewCheckpointInfo({ threadId }) ?? {}

		// Only add checkpoint if there are actual changes
		if (diffBasedCheckpointsOfURI && Object.keys(diffBasedCheckpointsOfURI).length > 0) {
			this._addCheckpoint(threadId, {
				role: 'checkpoint',
				type: 'user_edit',
				diffBasedCheckpointsOfURI: diffBasedCheckpointsOfURI ?? {},
				previousCheckpointIdx: previousCheckpointIdx,
				userModifications: { diffBasedCheckpointsOfURI: {}, },
			})
		}

		// Clear tracking after checkpointing (even if no changes found, we've processed them)
		this._clearUserChanges(threadId)
	}
	// call this right after LLM edits a file
	private _addToolEditCheckpoint({ threadId, uri, }: { threadId: string, uri: URI }) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return
		const { model } = this._voidModelService.getModel(uri)
		if (!model) return // should never happen

		// Find the last checkpoint to compute diff from
		const lastCheckpointIdx = findLastIdx(thread.messages, (m) => m.role === 'checkpoint') ?? -1
		let previousSnapshot: VoidFileSnapshot | null = null

		if (lastCheckpointIdx !== -1) {
			const lastCheckpoint = thread.messages[lastCheckpointIdx]
			if (lastCheckpoint.role === 'checkpoint') {
				// Try to get previous content from the last checkpoint
				const prevDiffCheckpoint = lastCheckpoint.diffBasedCheckpointsOfURI?.[uri.fsPath];
				if (prevDiffCheckpoint) {
					// For tool edits, we always store full snapshot since it's a single file
					// and we need to ensure we have the complete state for restoration
					previousSnapshot = {
						snapshottedDiffAreaOfId: prevDiffCheckpoint.snapshottedDiffAreaOfId,
						entireFileCode: prevDiffCheckpoint.isFullSnapshot
							? prevDiffCheckpoint.fileContentDiffs[0]?.newText || model.getValue()
							: model.getValue() // Fallback
					};
				} else if (lastCheckpoint.voidFileSnapshotOfURI?.[uri.fsPath]) {
					// Fall back to legacy format
					previousSnapshot = lastCheckpoint.voidFileSnapshotOfURI[uri.fsPath]!;
				}
			}
		}

		const currentSnapshot = this._editCodeService.getVoidFileSnapshot(uri)
		const diffCheckpoint = createDiffBasedCheckpoint(previousSnapshot, currentSnapshot)

		this._addCheckpoint(threadId, {
			role: 'checkpoint',
			type: 'tool_edit',
			diffBasedCheckpointsOfURI: { [uri.fsPath]: diffCheckpoint },
			previousCheckpointIdx: lastCheckpointIdx >= 0 ? lastCheckpointIdx : null,
			userModifications: { diffBasedCheckpointsOfURI: {} },
		})
	}


	private _getCheckpointBeforeMessage = ({ threadId, messageIdx }: { threadId: string, messageIdx: number }): [CheckpointEntry, number] | undefined => {
		const thread = this.state.allThreads[threadId]
		if (!thread) return undefined
		for (let i = messageIdx; i >= 0; i--) {
			const message = thread.messages[i]
			if (message.role === 'checkpoint') {
				return [message, i]
			}
		}
		return undefined
	}

	private _getCheckpointsBetween({ threadId, loIdx, hiIdx }: { threadId: string, loIdx: number, hiIdx: number }) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return { lastIdxOfURI: {} } // should never happen
		const lastIdxOfURI: { [fsPath: string]: number } = {}
		for (let i = loIdx; i <= hiIdx; i += 1) {
			const message = thread.messages[i]
			if (message?.role !== 'checkpoint') continue
			// Check new diff-based format first
			if (message.diffBasedCheckpointsOfURI) {
				for (const fsPath in message.diffBasedCheckpointsOfURI) {
					lastIdxOfURI[fsPath] = i
				}
			}
			// Fall back to legacy format
			if (message.voidFileSnapshotOfURI) {
				for (const fsPath in message.voidFileSnapshotOfURI) {
					lastIdxOfURI[fsPath] = i
				}
			}
		}
		return { lastIdxOfURI }
	}

	private _readCurrentCheckpoint(threadId: string): [CheckpointEntry, number] | undefined {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		const { currCheckpointIdx } = thread.state
		if (currCheckpointIdx === null) return

		const checkpoint = thread.messages[currCheckpointIdx]
		if (!checkpoint) return
		if (checkpoint.role !== 'checkpoint') return
		return [checkpoint, currCheckpointIdx]
	}
	private _addUserModificationsToCurrCheckpoint({ threadId }: { threadId: string }) {
		const { diffBasedCheckpointsOfURI } = this._computeNewCheckpointInfo({ threadId }) ?? {}
		const res = this._readCurrentCheckpoint(threadId)
		if (!res) return
		const [checkpoint, checkpointIdx] = res
		this._editMessageInThread(threadId, checkpointIdx, {
			...checkpoint,
			userModifications: { diffBasedCheckpointsOfURI: diffBasedCheckpointsOfURI ?? {}, },
		})
	}


	private _makeUsStandOnCheckpoint({ threadId }: { threadId: string }) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return
		if (thread.state.currCheckpointIdx === null) {
			const lastMsg = thread.messages[thread.messages.length - 1]
			if (lastMsg?.role !== 'checkpoint')
				this._addUserCheckpoint({ threadId })
			this._setThreadState(threadId, { currCheckpointIdx: thread.messages.length - 1 })
		}
	}

	jumpToCheckpointBeforeMessageIdx({ threadId, messageIdx, jumpToUserModified }: { threadId: string, messageIdx: number, jumpToUserModified: boolean }) {

		// if null, add a new temp checkpoint so user can jump forward again
		this._makeUsStandOnCheckpoint({ threadId })

		const thread = this.state.allThreads[threadId]
		if (!thread) return
		if (this.streamState[threadId]?.isRunning) return

		const c = this._getCheckpointBeforeMessage({ threadId, messageIdx })
		if (c === undefined) return // should never happen

		const fromIdx = thread.state.currCheckpointIdx
		if (fromIdx === null) return // should never happen

		const [_, toIdx] = c
		if (toIdx === fromIdx) return

		// voidDevLog(`going from ${fromIdx} to ${toIdx}`)

		// update the user's checkpoint
		this._addUserModificationsToCurrCheckpoint({ threadId })

		/*
	if undoing

	A,B,C are all files.
	x means a checkpoint where the file changed.

	A B C D E F G H I
	x x x x x   x           <-- you can't always go up to find the "before" version; sometimes you need to go down
	| | | | |   | x
	--x-|-|-|-x---x-|-----     <-- to
	| | | | x   x
	| | x x |
	| |   | |
	----x-|---x-x-------     <-- from
	  x

	We need to revert anything that happened between to+1 and from.
	**We do this by finding the last x from 0...`to` for each file and applying those contents.**
	We only need to do it for files that were edited since `to`, ie files between to+1...from.
	*/
		if (toIdx < fromIdx) {
			const { lastIdxOfURI } = this._getCheckpointsBetween({ threadId, loIdx: toIdx + 1, hiIdx: fromIdx })

			const idxes = function* () {
				for (let k = toIdx; k >= 0; k -= 1) { // first go up
					yield k
				}
				for (let k = toIdx + 1; k < thread.messages.length; k += 1) { // then go down
					yield k
				}
			}

			for (const fsPath in lastIdxOfURI) {
				// find the first instance of this file starting at toIdx (go up to latest file; if there is none, go down)
				for (const k of idxes()) {
					const message = thread.messages[k]
					if (message.role !== 'checkpoint') continue
					const res = this._getCheckpointInfo(message, fsPath, { includeUserModifiedChanges: jumpToUserModified })
					if (!res) continue
					const { voidFileSnapshot } = res
					if (!voidFileSnapshot) continue
					this._editCodeService.restoreVoidFileSnapshot(URI.file(fsPath), voidFileSnapshot)
					break
				}
			}
		}

		/*
	if redoing

	A B C D E F G H I J
	x x x x x   x     x
	| | | | |   | x x x
	--x-|-|-|-x---x-|-|---     <-- from
	| | | | x   x
	| | x x |
	| |   | |
	----x-|---x-x-----|---     <-- to
	  x           x


	We need to apply latest change for anything that happened between from+1 and to.
	We only need to do it for files that were edited since `from`, ie files between from+1...to.
	*/
		if (toIdx > fromIdx) {
			const { lastIdxOfURI } = this._getCheckpointsBetween({ threadId, loIdx: fromIdx + 1, hiIdx: toIdx })
			for (const fsPath in lastIdxOfURI) {
				// apply lowest down content for each uri
				for (let k = toIdx; k >= fromIdx + 1; k -= 1) {
					const message = thread.messages[k]
					if (message.role !== 'checkpoint') continue
					const res = this._getCheckpointInfo(message, fsPath, { includeUserModifiedChanges: jumpToUserModified })
					if (!res) continue
					const { voidFileSnapshot } = res
					if (!voidFileSnapshot) continue
					this._editCodeService.restoreVoidFileSnapshot(URI.file(fsPath), voidFileSnapshot)
					break
				}
			}
		}

		this._setThreadState(threadId, { currCheckpointIdx: toIdx })
	}


	private _wrapRunAgentToNotify(p: Promise<void>, threadId: string) {
		const notify = ({ error }: { error: string | null }) => {
			const thread = this.state.allThreads[threadId]
			if (!thread) return
			const userMsg = findLast(thread.messages, m => m.role === 'user')
			if (!userMsg) return
			if (userMsg.role !== 'user') return
			const messageContent = truncate(userMsg.displayContent, 50, '...')

			this._notificationService.notify({
				severity: error ? Severity.Warning : Severity.Info,
				message: error ? `Error: ${error} ` : `A new Chat result is ready.`,
				source: messageContent,
				sticky: true,
				actions: {
					primary: [{
						id: 'void.goToChat',
						enabled: true,
						label: `Jump to Chat`,
						tooltip: '',
						class: undefined,
						run: () => {
							this.switchToThread(threadId)
							// scroll to bottom
							this.state.allThreads[threadId]?.state.mountedInfo?.whenMounted.then(m => {
								m.scrollToBottom()
							})
						}
					}]
				},
			})
		}

		p.then(() => {
			if (threadId !== this.state.currentThreadId) notify({ error: null })
			// Auto-compact old conversation history now that the turn is complete.
			return this._maybeAutoCompact(threadId)
		}).catch((e) => {
			if (threadId !== this.state.currentThreadId) notify({ error: getErrorMessage(e) })
			// Log but do not re-throw to avoid unhandled promise rejection
			console.error('[chatThreadService] _wrapRunAgentToNotify caught error:', e)
		})
	}

	dismissStreamError(threadId: string): void {
		this._setStreamState(threadId, undefined)
	}


	private async _addUserMessageAndStreamResponse({ userMessage, _chatSelections, images, threadId, _isFromQueue = false }: { userMessage: string, _chatSelections?: StagingSelectionItem[], images?: ImageAttachment[], threadId: string, _isFromQueue?: boolean }) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return // should never happen

		// Drop cached read-only tool results for this thread. Between turns the
		// user (or an external process) may have edited files on disk, so a
		// memoized `read_file`/`search_*` from the previous turn could be stale.
		// Within a single turn the cache still dedups repeated identical reads;
		// only cross-turn reads are forced fresh.
		delete this.toolResultCache[threadId]

		// Fire SessionStart hook once per session (on the first user message).
		// `additionalContext` from the hook is prepended to the user message so it
		// reaches the model this turn. Non-blocking: hook errors are swallowed.
		if (!this._sessionStartFired) {
			this._sessionStartFired = true
			try {
				const ssHook = await this._hookService.fireSessionStart(this._sessionStartSource)
				if (ssHook.additionalContext) {
					userMessage = `${ssHook.additionalContext}\n\n${userMessage}`
				}
			} catch (err) {
				voidDevWarn('[hooks] SessionStart fire threw (non-blocking):', err)
			}
		}

		// interrupt existing stream
		if (this.streamState[threadId]?.isRunning) {
			await this.abortRunning(threadId)
		}

		// add dummy before this message to keep checkpoint before user message idea consistent
		if (thread.messages.length === 0) {
			this._addUserCheckpoint({ threadId })
		}


		// add user's message to chat history
		const currSelns: StagingSelectionItem[] = _chatSelections ?? thread.state.stagingSelections

		// Process images FIRST if present (before adding message)
		let visionAnalysis: string | undefined;
		// MEMORY OPTIMIZATION: Validate and limit images to prevent memory bloat
		if (images && images.length > 0) {
			// Limit number of images
			if (images.length > MAX_IMAGES_PER_MESSAGE) {
				voidDevWarn(`[Memory] Limiting images from ${images.length} to ${MAX_IMAGES_PER_MESSAGE}`);
				images = images.slice(0, MAX_IMAGES_PER_MESSAGE);
			}

			// Check total image size
			const totalSizeMB = images.reduce((sum, img) => sum + (img.base64?.length || 0) * 0.75 / 1024 / 1024, 0);
			if (totalSizeMB > MAX_TOTAL_IMAGE_SIZE_MB) {
				voidDevWarn(`[Memory] Total image size ${totalSizeMB.toFixed(2)}MB exceeds limit of ${MAX_TOTAL_IMAGE_SIZE_MB}MB`);
				this._notificationService.notify({
					severity: Severity.Warning,
					message: `Images too large (${totalSizeMB.toFixed(1)}MB). Please use smaller images or fewer images.`,
				});
				images = []; // Clear images if too large
			}

			if (images.length > 0 && this._settingsService.state.globalSettings.enableVisionSupport) {
				// Show typing indicator while processing images
				this._setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed' });

				try {
					visionAnalysis = await this._visionService.processImages(images, userMessage);
				} catch (error) {
					console.error(`[chatThreadService] Error processing images:`, error);
					this._notificationService.notify({
						severity: Severity.Warning,
						message: `Failed to process images: ${error instanceof Error ? error.message : 'Unknown error'}`,
					});
				} finally {
					// Clear stream state after processing
					this._setStreamState(threadId, undefined);
				}
			}

			// MEMORY OPTIMIZATION: Clear base64 image data after processing to prevent memory bloat
			// The visionAnalysis text is preserved, but the large base64 data is discarded
			if (images.length > 0) {
				images = images.map(img => ({
					...img,
					base64: '[processed]' // Replace base64 with placeholder
				}));
			}
		}

		// Build message content with vision analysis if available
		const messageContent = visionAnalysis
			? (userMessage ? `${userMessage}\n\n[Image Analysis]\n${visionAnalysis}` : `[Image Analysis]\n${visionAnalysis}`)
			: userMessage;

		let finalContent = await chat_userMessageContent(messageContent, currSelns, { directoryStrService: this._directoryStringService, fileService: this._fileService })

		// Fire UserPromptSubmit hook. A `block` decision prevents the user message
		// from entering the thread / starting the agent (the reason is surfaced to
		// the UI as a stream error). `additionalContext` is appended to the user
		// message content so it reaches the model this turn.
		try {
			const upHook = await this._hookService.fireUserPromptSubmit(threadId, userMessage)
			if (upHook.decision === 'block') {
				const blockReason = upHook.reason || 'Your prompt was blocked by a UserPromptSubmit hook.'
				this._setStreamState(threadId, { isRunning: undefined, error: { message: blockReason, fullError: null } })
				return
			}
			if (upHook.additionalContext) {
				finalContent = `${finalContent}\n\n${upHook.additionalContext}`
			}
		} catch (err) {
			voidDevWarn('[hooks] UserPromptSubmit fire threw (non-blocking):', err)
		}

		// Tool Orchestration: Get tool suggestions before adding user message to thread
		let orchestrationResult: OrchestrationResult = { suggestions: [], reasoning: '', summary: '' };
		const chatMode = this._settingsService.state.globalSettings.chatMode;

		// NEW: Auto-create workflow for complex requests in code mode
		if (chatMode === 'code' && userMessage.length > 50 && !_isFromQueue) {
			const isComplexRequest = this._detectComplexRequest(userMessage);
			if (isComplexRequest && !thread.state.activeWorkflow) {
				voidDevLog('[chatThreadService] Complex request detected, initializing workflow...');

				// Create active workflow
				thread.state.activeWorkflow = {
					id: generateUuid(),
					goal: userMessage,
					tasks: [], // Will be populated by LLM via create_plan
					currentTaskId: null,
					status: 'planning',
					createdAt: Date.now()
				};

				thread.state.queueBehavior = 'wait_for_workflow';
				this._storeAllThreads(this.state.allThreads);
				this._onDidChangeCurrentThread.fire();
			}
		}

		if (this._settingsService.state.globalSettings.enableToolOrchestration) {
			voidDevLog('[chatThreadService] Running tool orchestration...');
			this._setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed' });
			try {
				orchestrationResult = await this._orchestrationService.orchestrate({
					userMessage: userMessage,
					chatMode,
					onProgress: (reasoning) => {
						// Could show progress in UI if needed
					},
				});
				voidDevLog('[chatThreadService] Orchestration result:', orchestrationResult);
			} catch (error) {
				console.error('[chatThreadService] Orchestration error:', error);
				orchestrationResult = { suggestions: [], reasoning: '', summary: '' };
			} finally {
				this._setStreamState(threadId, undefined);
			}
		}

		const userHistoryElt: ChatMessage = {
			role: 'user',
			content: finalContent,
			displayContent: userMessage,
			selections: currSelns,
			images,
			visionAnalysis,
			state: defaultMessageState,
			// Store orchestration result for use in LLM prompt
			orchestrationResult: orchestrationResult.suggestions.length > 0 ? orchestrationResult : undefined,
		}
		this._addMessageToThread(threadId, userHistoryElt)

		this._setThreadState(threadId, { currCheckpointIdx: null }) // no longer at a checkpoint because started streaming

		this._wrapRunAgentToNotify(
			this._runChatAgent({ threadId, ...this._currentModelSelectionProps(), orchestrationResult: orchestrationResult.suggestions.length > 0 ? orchestrationResult : undefined }),
			threadId,
		)

		// scroll to bottom
		this.state.allThreads[threadId]?.state.mountedInfo?.whenMounted.then(m => {
			m.scrollToBottom()
		})
	}


	async addUserMessageAndStreamResponse({ userMessage, selections, images, threadId }: { userMessage: string, selections?: StagingSelectionItem[], images?: ImageAttachment[], threadId: string }) {
		const thread = this.state.allThreads[threadId];
		if (!thread) return

		// Check if the thread is currently running
		const isRunning = this.streamState[threadId]?.isRunning;

		// NEW: Override workflow for manual send (not from queue)
		if (!isRunning && !this._hasQueuedMessages(threadId)) {
			this._overrideWorkflow(threadId);
		}

		// If the thread is running, queue the message instead of aborting
		if (isRunning) {
			voidDevLog(`[chatThreadService] Thread ${threadId} is currently running. Queueing message.`);
			this._queueMessage(threadId, { userMessage, selections, images });
			return;
		}

		// Now call the original method to add the user message and stream the response
		await this._addUserMessageAndStreamResponse({ userMessage, _chatSelections: selections, images, threadId });
	}

	editUserMessageAndStreamResponse: IChatThreadService['editUserMessageAndStreamResponse'] = async ({ userMessage, messageIdx, threadId }) => {

		const thread = this.state.allThreads[threadId]
		if (!thread) return // should never happen

		if (thread.messages?.[messageIdx]?.role !== 'user') {
			throw new Error(`Error: editing a message with role !=='user'`)
		}

		// get prev and curr selections before clearing the message
		const currSelns = thread.messages[messageIdx].state.stagingSelections || [] // staging selections for the edited message

		// clear messages up to the index
		const slicedMessages = thread.messages.slice(0, messageIdx)
		this._setState({
			allThreads: {
				...this.state.allThreads,
				[thread.id]: {
					...thread,
					messages: slicedMessages
				}
			}
		})

		// re-add the message and stream it
		await this._addUserMessageAndStreamResponse({ userMessage, _chatSelections: currSelns, threadId })
	}

	// ---------- Message Queue Methods ----------

	private _queueMessage(threadId: string, message: { userMessage: string, selections?: StagingSelectionItem[], images?: ImageAttachment[] }) {
		if (!this.messageQueue[threadId]) {
			this.messageQueue[threadId] = [];
		}
		// Deduplication: skip if identical to the last queued or last sent message
		const queue = this.messageQueue[threadId];
		const trimmed = message.userMessage.trim();
		if (trimmed && queue.length > 0 && queue[queue.length - 1].userMessage.trim() === trimmed) {
			voidDevLog(`[chatThreadService] Skipping duplicate queued message for thread ${threadId}`);
			return;
		}
		// Also check against the last user message in the thread
		const threadMessages = this.state.allThreads[threadId]?.messages;
		if (trimmed && threadMessages) {
			const lastUserMsg = [...threadMessages].reverse().find(m => m.role === 'user');
			if (lastUserMsg?.content?.trim() === trimmed) {
				voidDevLog(`[chatThreadService] Skipping duplicate queued message (matches last sent) for thread ${threadId}`);
				return;
			}
		}
		// MEMORY OPTIMIZATION: Limit queue size to prevent unbounded memory growth.
		// When full, drop the OLDEST message to make room, then queue the new one below.
		if (this.messageQueue[threadId].length >= MAX_MESSAGE_QUEUE_PER_THREAD) {
			voidDevWarn(`[Memory] Message queue for thread ${threadId} is full (${MAX_MESSAGE_QUEUE_PER_THREAD}). Dropping oldest message to make room.`);
			this.messageQueue[threadId].shift(); // Remove oldest message to make room
		}
		this.messageQueue[threadId].push(message);
		voidDevLog(`[chatThreadService] Queued message for thread ${threadId}. Queue length: ${this.messageQueue[threadId].length}`);
		// Fire dedicated queue event to update UI
		this._onDidChangeMessageQueue.fire({ threadId });
		// Trigger processing logic if we're idle
		if (!this.streamState[threadId]?.isRunning) {
			this._processNextQueuedMessage(threadId)
		}
	}

	private _hasQueuedMessages(threadId: string): boolean {
		return !!(this.messageQueue[threadId] && this.messageQueue[threadId].length > 0);
	}

	private async _processNextQueuedMessage(threadId: string) {
		if (!this._hasQueuedMessages(threadId)) {
			// ensure UI updates if queue emptied
			this._onDidChangeMessageQueue.fire({ threadId });
			return;
		}
		if (this.streamState[threadId]?.isRunning) {
			return;
		}

		const nextMessage = this.messageQueue[threadId].shift();
		if (!nextMessage) return;

		voidDevLog(`[chatThreadService] Processing queued message. Remaining in queue: ${this.messageQueue[threadId].length}`);
		// Fire dedicated queue event to update UI
		this._onDidChangeMessageQueue.fire({ threadId });

		// Small delay to ensure UI updates
		await timeout(100);

		// Process the queued message
		await this._addUserMessageAndStreamResponse({
			userMessage: nextMessage.userMessage,
			_chatSelections: nextMessage.selections,
			images: nextMessage.images,
			threadId,
			_isFromQueue: true,  // NEW: Mark as from queue
		});
	}

	getQueuedMessagesCount(threadId: string): number {
		return this.messageQueue[threadId]?.length || 0;
	}

	getQueuedMessages(threadId: string): Array<{ userMessage: string, selections?: StagingSelectionItem[], images?: ImageAttachment[] }> {
		return this.messageQueue[threadId] || [];
	}

	removeQueuedMessage(threadId: string, index: number): void {
		if (this.messageQueue[threadId] && this.messageQueue[threadId][index]) {
			this.messageQueue[threadId].splice(index, 1);
			voidDevLog(`[chatThreadService] Removed queued message at index ${index}. Remaining: ${this.messageQueue[threadId].length}`);
			this._onDidChangeMessageQueue.fire({ threadId });
		}
	}

	clearMessageQueue(threadId: string) {
		if (this.messageQueue[threadId]) {
			this.messageQueue[threadId] = [];
			voidDevLog(`[chatThreadService] Cleared message queue for thread ${threadId}`);
			this._onDidChangeMessageQueue.fire({ threadId });
		}
	}

	async forceSendQueuedMessage(threadId: string, index: number): Promise<void> {
		const message = this.messageQueue[threadId]?.[index];
		if (!message) return;

		// Remove from queue
		this.messageQueue[threadId].splice(index, 1);
		voidDevLog(`[chatThreadService] Force sending queued message at index ${index}`);
		this._onDidChangeMessageQueue.fire({ threadId });

		// Abort current LLM if running — wait for the interrupt promise to resolve
		const streamState = this.streamState[threadId];
		if (streamState?.isRunning && streamState.interrupt !== 'not_needed') {
				const interruptFn = await streamState.interrupt;
				if (typeof interruptFn === 'function') {
					interruptFn();
				}
				// Grace period for stream state cleanup after abort
		}

		// Send the message
		await this._addUserMessageAndStreamResponse({
			userMessage: message.userMessage,
			_chatSelections: message.selections,
			images: message.images,
			threadId
		});
	}

	/**
	 * Override the current active workflow (called when user manually sends message)
	 */
	private _overrideWorkflow(threadId: string): void {
		const thread = this.state.allThreads[threadId];
		if (!thread) return;

		if (thread.state.activeWorkflow) {
			voidDevLog('[chatThreadService] Workflow override requested, clearing workflow');

			// Mark workflow as cancelled/failed
			thread.state.activeWorkflow = null;

			// Clear task plan
			this.clearTaskPlan(threadId);

			// Reset queue behavior
			this._updateThreadStateAndStore(threadId, { queueBehavior: 'wait_for_workflow' });

			this._onDidChangeCurrentThread.fire();
		}
	}

	/**
	 * Detect if a user request is complex enough to warrant workflow planning
	 */
	private _detectComplexRequest(message: string): boolean {
		const complexPatterns = [
			/redesign|refactor|implement|build.*system|create.*feature|add.*system/i,
			/multiple.*files|several.*pages|all.*components/i,
			/step\s+\d|first.*then|after.*that/i,
			/and.*also|and.*then|additionally/i,
			/complete.*system|full.*implementation/i,
		];

		return complexPatterns.some(pattern => pattern.test(message));
	}

	// ---------- the rest ----------

	private _getAllSeenFileURIs(threadId: string) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return []

		const fsPathsSet = new Set<string>()
		const uris: URI[] = []
		const addURI = (uri: URI) => {
			if (!fsPathsSet.has(uri.fsPath)) {
				uris.push(uri)
				fsPathsSet.add(uri.fsPath)
			}
		}

		for (const m of thread.messages) {
			// URIs of user selections
			if (m.role === 'user') {
				for (const sel of m.selections ?? []) {
					addURI(sel.uri)
				}
			}
			// URIs of files that have been read
			else if (m.role === 'tool' && m.type === 'success' && m.name === 'read_file') {
				const params = m.params as BuiltinToolCallParams['read_file']
				addURI(params.uri)
			}
		}
		return uris
	}



	getRelativeStr = (uri: URI) => {
		const isInside = this._workspaceContextService.isInsideWorkspace(uri)
		if (isInside) {
			const f = this._workspaceContextService.getWorkspace().folders.find(f => uri.fsPath.startsWith(f.uri.fsPath))
			if (f) { return uri.fsPath.replace(f.uri.fsPath, '') }
			else { return undefined }
		}
		else {
			return undefined
		}
	}


	// gets the location of codespan link so the user can click on it
	generateCodespanLink: IChatThreadService['generateCodespanLink'] = async ({ codespanStr: _codespanStr, threadId }) => {

		// process codespan to understand what we are searching for
		// Simplified pattern: supports alphanumeric identifiers; could extend for method signatures
		const functionOrMethodPattern = /^[a-zA-Z_$][a-zA-Z0-9_$]*$/; // `fUnCt10n_name`
		const functionParensPattern = /^([^\s(]+)\([^)]*\)$/; // `functionName( args )`

		let target = _codespanStr // the string to search for
		let codespanType: 'file-or-folder' | 'function-or-class'
		if (target.includes('.') || target.includes('/')) {

			codespanType = 'file-or-folder'
			target = _codespanStr

		} else if (functionOrMethodPattern.test(target)) {

			codespanType = 'function-or-class'
			target = _codespanStr

		} else if (functionParensPattern.test(target)) {
			const match = target.match(functionParensPattern)
			if (match && match[1]) {

				codespanType = 'function-or-class'
				target = match[1]

			}
			else { return null }
		}
		else {
			return null
		}

		// get history of all AI and user added files in conversation + store in reverse order (MRU)
		const prevUris = this._getAllSeenFileURIs(threadId).reverse()

		if (codespanType === 'file-or-folder') {
			const doesUriMatchTarget = (uri: URI) => uri.path.includes(target)

			// check if any prevFiles are the `target`
			for (const [idx, uri] of prevUris.entries()) {
				if (doesUriMatchTarget(uri)) {

					// shorten it

					// URI display text shortening for this occurrence
					const prevUriStrs = prevUris.map(uri => uri.fsPath)
					const shortenedUriStrs = shorten(prevUriStrs)
					let displayText = shortenedUriStrs[idx]
					const ellipsisIdx = displayText.lastIndexOf('…/');
					if (ellipsisIdx >= 0) {
						displayText = displayText.slice(ellipsisIdx + 2)
					}

					return { uri, displayText }
				}
			}

			// else search codebase for `target`
			let uris: URI[] = []
			try {
				const { result } = await this._toolsService.callTool['search_pathnames_only']({ query: target, includePattern: null, pageNumber: 0 })
				const { uris: uris_ } = await result
				uris = uris_
			} catch (e) {
				return null
			}

			for (const [idx, uri] of uris.entries()) {
				if (doesUriMatchTarget(uri)) {

					// URI display text shortening for repeated occurrences
					const prevUriStrs = prevUris.map(uri => uri.fsPath)
					const shortenedUriStrs = shorten(prevUriStrs)
					let displayText = shortenedUriStrs[idx]
					const ellipsisIdx = displayText.lastIndexOf('…/');
					if (ellipsisIdx >= 0) {
						displayText = displayText.slice(ellipsisIdx + 2)
					}


					return { uri, displayText }
				}
			}

		}


		if (codespanType === 'function-or-class') {


			// check all prevUris for the target
			for (const uri of prevUris) {

				const modelRef = await this._voidModelService.getModelSafe(uri)
				const { model } = modelRef
				if (!model) continue

				const matches = model.findMatches(
					target,
					false, // searchOnlyEditableRange
					false, // isRegex
					true,  // matchCase
					null, //' ',   // wordSeparators
					true   // captureMatches
				);

				const firstThree = matches.slice(0, 3);

				// take first 3 occurences, attempt to goto definition on them
				for (const match of firstThree) {
					const position = new Position(match.range.startLineNumber, match.range.startColumn);
					const definitionProviders = this._languageFeaturesService.definitionProvider.ordered(model);

					for (const provider of definitionProviders) {

						const _definitions = await provider.provideDefinition(model, position, CancellationToken.None);

						if (!_definitions) continue;

						const definitions = Array.isArray(_definitions) ? _definitions : [_definitions];

						for (const definition of definitions) {

							return {
								uri: definition.uri,
								selection: {
									startLineNumber: definition.range.startLineNumber,
									startColumn: definition.range.startColumn,
									endLineNumber: definition.range.endLineNumber,
									endColumn: definition.range.endColumn,
								},
								displayText: _codespanStr,
							};

							// const defModelRef = await this._textModelService.createModelReference(definition.uri);
							// const defModel = defModelRef.object.textEditorModel;

							// try {
							// 	const symbolProviders = this._languageFeaturesService.documentSymbolProvider.ordered(defModel);

							// 	for (const symbolProvider of symbolProviders) {
							// 		const symbols = await symbolProvider.provideDocumentSymbols(
							// 			defModel,
							// 			CancellationToken.None
							// 		);

							// 		if (symbols) {
							// 			const symbol = symbols.find(s => {
							// 				const symbolRange = s.range;
							// 				return symbolRange.startLineNumber <= definition.range.startLineNumber &&
							// 					symbolRange.endLineNumber >= definition.range.endLineNumber &&
							// 					(symbolRange.startLineNumber !== definition.range.startLineNumber || symbolRange.startColumn <= definition.range.startColumn) &&
							// 					(symbolRange.endLineNumber !== definition.range.endLineNumber || symbolRange.endColumn >= definition.range.endColumn);
							// 			});

							// 			// if we got to a class/function get the full range and return
							// 			if (symbol?.kind === SymbolKind.Function || symbol?.kind === SymbolKind.Method || symbol?.kind === SymbolKind.Class) {
							// 				return {
							// 					uri: definition.uri,
							// 					selection: {
							// 						startLineNumber: definition.range.startLineNumber,
							// 						startColumn: definition.range.startColumn,
							// 						endLineNumber: definition.range.endLineNumber,
							// 						endColumn: definition.range.endColumn,
							// 					}
							// 				};
							// 			}
							// 		}
							// 	}
							// } finally {
							// 	defModelRef.dispose();
							// }
						}
					}
				}
			}

			// unlike above do not search codebase (doesnt make sense)

		}

		return null

	}

	getCodespanLink({ codespanStr, messageIdx, threadId }: { codespanStr: string, messageIdx: number, threadId: string }): CodespanLocationLink | undefined {
		const thread = this.state.allThreads[threadId]
		if (!thread) return undefined;

		const links = thread.state.linksOfMessageIdx?.[messageIdx]
		if (!links) return undefined;

		const link = links[codespanStr]

		return link
	}

	async addCodespanLink({ newLinkText, newLinkLocation, messageIdx, threadId }: { newLinkText: string, newLinkLocation: CodespanLocationLink, messageIdx: number, threadId: string }) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		this._setState({

			allThreads: {
				...this.state.allThreads,
				[threadId]: {
					...thread,
					state: {
						...thread.state,
						linksOfMessageIdx: {
							...thread.state.linksOfMessageIdx,
							[messageIdx]: {
								...thread.state.linksOfMessageIdx?.[messageIdx],
								[newLinkText]: newLinkLocation
							}
						}
					}

				}
			}
		})
	}


	getCurrentThread(): ThreadType {
		const state = this.state
		const thread = state.allThreads[state.currentThreadId]
		if (!thread) throw new Error(`Current thread should never be undefined`)
		return thread
	}

	getCurrentFocusedMessageIdx() {
		const thread = this.getCurrentThread()

		// get the focusedMessageIdx
		const focusedMessageIdx = thread.state.focusedMessageIdx
		if (focusedMessageIdx === undefined) return;

		// check that the message is actually being edited
		const focusedMessage = thread.messages[focusedMessageIdx]
		if (focusedMessage.role !== 'user') return;
		if (!focusedMessage.state) return;

		return focusedMessageIdx
	}

	isCurrentlyFocusingMessage() {
		return this.getCurrentFocusedMessageIdx() !== undefined
	}

	switchToThread(threadId: string) {
		this._setState({ currentThreadId: threadId })
	}


	openNewThread() {
		// if a thread with 0 messages already exists, switch to it
		const { allThreads: currentThreads } = this.state
		for (const threadId in currentThreads) {
			if (currentThreads[threadId]!.messages.length === 0) {
				// switch to the existing empty thread and exit
				this.switchToThread(threadId)
				return
			}
		}
		// otherwise, start a new thread
		const newThread = newThreadObject()

		// update state
		const newThreads: ChatThreads = {
			...currentThreads,
			[newThread.id]: newThread
		}
		this._storeAllThreads(newThreads)
		this._setState({ allThreads: newThreads, currentThreadId: newThread.id })
	}


	deleteThread(threadId: string): void {
		const { allThreads: currentThreads } = this.state

		// delete the thread
		const newThreads = { ...currentThreads };
		delete newThreads[threadId];

		// MEMORY FIX: Clean up associated data structures to prevent memory leaks
		delete this.toolCallHistory[threadId];
		delete this.toolResultCache[threadId];
		delete this.messageQueue[threadId];
		delete this.taskPlans[threadId];
		delete this.streamState[threadId];
		this.usageAnchorByThreadId.delete(threadId);
		// Drop this thread's plans so the per-thread plan maps don't leak.
		this._toolsService.getPlanningService().clearPlan(threadId)
		this._toolsService.getImplementationPlanningService().clearPlan(threadId)

		// store the updated threads
		this._storeAllThreads(newThreads);
		this._setState({ ...this.state, allThreads: newThreads })
	}

	duplicateThread(threadId: string) {
		const { allThreads: currentThreads } = this.state
		const threadToDuplicate = currentThreads[threadId]
		if (!threadToDuplicate) return
		const newThread = {
			...deepClone(threadToDuplicate),
			id: generateUuid(),
		}
		const newThreads = {
			...currentThreads,
			[newThread.id]: newThread,
		}
		this._storeAllThreads(newThreads)
		this._setState({ allThreads: newThreads })
	}

	setThreadName(threadId: string, name: string) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return
		const newThreads = {
			...this.state.allThreads,
			[threadId]: {
				...thread,
				name,
				lastModified: new Date().toISOString(),
			}
		}
		this._storeAllThreads(newThreads)
		this._setState({ allThreads: newThreads })
	}


	/**
	 * Calculate approximate size of a message in KB
	 * Includes content, tool results, and reasoning
	 */
	private _calculateMessageSizeKB(message: ChatMessage): number {
		let sizeBytes = 0;

		// Content size
		if ('displayContent' in message && message.displayContent) {
			sizeBytes += new Blob([message.displayContent]).size;
		}
		if ('content' in message && message.content) {
			sizeBytes += new Blob([message.content]).size;
		}
		if ('reasoning' in message && message.reasoning) {
			sizeBytes += new Blob([message.reasoning]).size;
		}

		// Tool result size
		if ('result' in message && message.result) {
			const resultStr = typeof message.result === 'string'
				? message.result
				: JSON.stringify(message.result);
			sizeBytes += new Blob([resultStr]).size;
		}

		// Tool params size
		if ('params' in message && message.params) {
			const paramsStr = JSON.stringify(message.params);
			sizeBytes += new Blob([paramsStr]).size;
		}

		// Convert to KB
		return sizeBytes / 1024;
	}

	/**
	 * Calculate total thread size in MB
	 */
	private _calculateThreadSizeMB(messages: ChatMessage[]): number {
		let totalBytes = 0;

		for (const message of messages) {
			// Content size
			if ('displayContent' in message && message.displayContent) {
				totalBytes += new Blob([message.displayContent]).size;
			}
			if ('content' in message && message.content) {
				totalBytes += new Blob([message.content]).size;
			}
			if ('reasoning' in message && message.reasoning) {
				totalBytes += new Blob([message.reasoning]).size;
			}

			// Tool result size
			if ('result' in message && message.result) {
				const resultStr = typeof message.result === 'string'
					? message.result
					: JSON.stringify(message.result);
				totalBytes += new Blob([resultStr]).size;
			}

			// Tool params size
			if ('params' in message && message.params) {
				const paramsStr = JSON.stringify(message.params);
				totalBytes += new Blob([paramsStr]).size;
			}
		}

		// Convert to MB
		return totalBytes / (1024 * 1024);
	}

	/**
	 * Truncate tool results in a message to fit size limits
	 */
	private _truncateMessageToFitSizeLimit(message: ChatMessage, maxSizeKB: number): ChatMessage {
		const currentSize = this._calculateMessageSizeKB(message);
		if (currentSize <= maxSizeKB) return message;

		const truncatedMessage = { ...message };

		// Truncate tool results first (largest contributor)
		if ('result' in truncatedMessage && truncatedMessage.result) {
			const resultStr = typeof truncatedMessage.result === 'string'
				? truncatedMessage.result
				: JSON.stringify(truncatedMessage.result);

			if (resultStr.length > 100) {
				const truncatedResult = resultStr.substring(0, 100) + `\n\n[... truncated for memory management - original size: ${(resultStr.length / 1024).toFixed(1)}KB]`;
				truncatedMessage.result = truncatedResult;

				voidDevLog(`[Memory] Truncated tool result from ${(resultStr.length / 1024).toFixed(1)}KB to ${(truncatedResult.length / 1024).toFixed(1)}KB`);
			}
		}

		// Truncate content if still over limit
		if ('displayContent' in truncatedMessage && truncatedMessage.displayContent) {
			const contentSize = new Blob([truncatedMessage.displayContent]).size / 1024;
			if (contentSize > maxSizeKB / 2) {
				const maxChars = Math.floor((maxSizeKB / 2) * 1024);
				if (truncatedMessage.displayContent.length > maxChars) {
					truncatedMessage.displayContent = truncatedMessage.displayContent.substring(0, maxChars) + `\n\n[... truncated for memory management]`;
					voidDevLog(`[Memory] Truncated displayContent from ${(contentSize).toFixed(1)}KB to ${(maxSizeKB / 2).toFixed(1)}KB`);
				}
			}
		}

		return truncatedMessage;
	}

	private _addMessageToThread(threadId: string, message: ChatMessage) {
		const { allThreads } = this.state
		const oldThread = allThreads[threadId]
		if (!oldThread) return // should never happen

		// Add timestamp if not present
		if (!('_timestamp' in message)) {
			message._timestamp = Date.now();
		}

		// SIZE-BASED LIMIT: Truncate message if it exceeds size limit before adding
		let processedMessage = message;
		const messageSizeKB = this._calculateMessageSizeKB(message);

		if (messageSizeKB > MAX_MESSAGE_SIZE_KB) {
			voidDevLog(`[Memory] Message size (${messageSizeKB.toFixed(1)}KB) exceeds limit (${MAX_MESSAGE_SIZE_KB}KB), truncating`);
			processedMessage = this._truncateMessageToFitSizeLimit(message, MAX_MESSAGE_SIZE_KB);
		}

		// MEMORY OPTIMIZATION: Prune old messages if exceeding max limit
		let messages = [...oldThread.messages, processedMessage];

		// Limit total messages
		if (messages.length > MAX_MESSAGES_PER_THREAD) {
			messages = messages.slice(-MAX_MESSAGES_PER_THREAD);
			voidDevLog(`[Memory] Pruned thread ${threadId} total messages to ${messages.length}`);
		}

		// SIZE-BASED LIMIT: Check total thread size and prune if necessary
		let threadSizeMB = this._calculateThreadSizeMB(messages);
		if (threadSizeMB > MAX_THREAD_SIZE_MB) {
			voidDevLog(`[Memory] Thread size (${threadSizeMB.toFixed(1)}MB) exceeds limit (${MAX_THREAD_SIZE_MB}MB), pruning old messages`);

			// Progressively remove oldest non-checkpoint messages until under limit
			let removeIdx = 0;
			while (threadSizeMB > MAX_THREAD_SIZE_MB && removeIdx < messages.length - 10) {
				// Skip checkpoints - they're important for state restoration
				if (messages[removeIdx].role === 'checkpoint') {
					removeIdx++;
					continue;
				}
				messages.splice(removeIdx, 1);
				threadSizeMB = this._calculateThreadSizeMB(messages);
				voidDevLog(`[Memory] Removed message at index ${removeIdx}, new size: ${threadSizeMB.toFixed(1)}MB`);
			}

			// If still over limit even after removing non-checkpoints, warn user
			if (threadSizeMB > MAX_THREAD_SIZE_MB) {
				voidDevWarn(`[Memory] Thread ${threadId} still exceeds size limit (${threadSizeMB.toFixed(1)}MB) after pruning. Consider starting a new thread.`);
			}
		}

		// MEMORY OPTIMIZATION: Limit number of checkpoints to prevent snapshot bloat
		const checkpointIndices = messages.reduce((acc, msg, idx) => {
			if (msg.role === 'checkpoint') acc.push(idx);
			return acc;
		}, [] as number[]);

		if (checkpointIndices.length > MAX_CHECKPOINTS_PER_THREAD) {
			const numToRemove = checkpointIndices.length - MAX_CHECKPOINTS_PER_THREAD;
			const indicesToRemove = new Set(checkpointIndices.slice(0, numToRemove));
			messages = messages.filter((_, idx) => !indicesToRemove.has(idx));
			voidDevLog(`[Memory] Pruned ${numToRemove} old checkpoints from thread ${threadId}`);
		}

		// update state and store it
		const newThreads = {
			...allThreads,
			[oldThread.id]: {
				...oldThread,
				lastModified: new Date().toISOString(),
				messages,
			},
		}
		this._storeAllThreads(newThreads)
		this._setState({ allThreads: newThreads }) // the current thread just changed (it had a message added to it)

		// Recalculate token usage after adding message
		this._updateTokenUsage(threadId)
	}

	/**
	 * Recalculate and update token usage for the current thread
	 */
	private async _updateTokenUsage(threadId: string) {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		const modelSelection = this._settingsService.state.modelSelectionOfFeature['Chat']
		if (!modelSelection) return

		const { chatMode } = this._settingsService.state.globalSettings

		try {
			const { tokenUsage } = await this._convertToLLMMessagesService.prepareLLMChatMessages({
				chatMessages: thread.messages,
				modelSelection,
				chatMode,
				loadedSkills: thread.state.loadedSkills,
				compaction: thread.state.compaction,
				usageAnchor: this.usageAnchorByThreadId.get(threadId),
			})

			// Update stream state with new token usage
			const currentState = this.streamState[threadId]
			if (currentState) {
				currentState.tokenUsage = tokenUsage
				this._onDidChangeStreamState.fire({ threadId })
			} else {
				// If no stream state, create one with just token usage
				this.streamState[threadId] = {
					isRunning: undefined,
					tokenUsage
				}
				this._onDidChangeStreamState.fire({ threadId })
			}
		} catch (error) {
			// Silently fail - token counting is not critical
			voidDevWarn('[chatThreadService] Failed to update token usage:', error)
		}
	}

	// sets the currently selected message (must be undefined if no message is selected)
	setCurrentlyFocusedMessageIdx(messageIdx: number | undefined) {

		const threadId = this.state.currentThreadId
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		this._setState({
			allThreads: {
				...this.state.allThreads,
				[threadId]: {
					...thread,
					state: {
						...thread.state,
						focusedMessageIdx: messageIdx,
					}
				}
			}
		})

		// // when change focused message idx, jump - do not jump back when click edit, too confusing.
		// if (messageIdx !== undefined)
		// 	this.jumpToCheckpointBeforeMessageIdx({ threadId, messageIdx, jumpToUserModified: true })
	}


	addNewStagingSelection(newSelection: StagingSelectionItem): void {

		const focusedMessageIdx = this.getCurrentFocusedMessageIdx()

		// set the selections to the proper value
		let selections: StagingSelectionItem[] = []
		let setSelections = (s: StagingSelectionItem[]) => { }

		if (focusedMessageIdx === undefined) {
			selections = this.getCurrentThreadState().stagingSelections
			setSelections = (s: StagingSelectionItem[]) => this.setCurrentThreadState({ stagingSelections: s })
		} else {
			selections = this.getCurrentMessageState(focusedMessageIdx).stagingSelections
			setSelections = (s) => this.setCurrentMessageState(focusedMessageIdx, { stagingSelections: s })
		}

		// if matches with existing selection, overwrite (since text may change)
		const idx = findStagingSelectionIndex(selections, newSelection)
		if (idx !== null && idx !== -1) {
			setSelections([
				...selections!.slice(0, idx),
				newSelection,
				...selections!.slice(idx + 1, Infinity)
			])
		}
		// if no match, add it
		else {
			setSelections([...(selections ?? []), newSelection])
		}
	}


	// Pops the staging selections from the current thread's state
	popStagingSelections(numPops: number): void {

		numPops = numPops ?? 1;

		const focusedMessageIdx = this.getCurrentFocusedMessageIdx()

		// set the selections to the proper value
		let selections: StagingSelectionItem[] = []
		let setSelections = (s: StagingSelectionItem[]) => { }

		if (focusedMessageIdx === undefined) {
			selections = this.getCurrentThreadState().stagingSelections
			setSelections = (s: StagingSelectionItem[]) => this.setCurrentThreadState({ stagingSelections: s })
		} else {
			selections = this.getCurrentMessageState(focusedMessageIdx).stagingSelections
			setSelections = (s) => this.setCurrentMessageState(focusedMessageIdx, { stagingSelections: s })
		}

		setSelections([
			...selections.slice(0, selections.length - numPops)
		])

	}

	// set message.state
	private _setCurrentMessageState(state: Partial<UserMessageState>, messageIdx: number): void {

		const threadId = this.state.currentThreadId
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		this._setState({
			allThreads: {
				...this.state.allThreads,
				[threadId]: {
					...thread,
					messages: thread.messages.map((m, i) =>
						i === messageIdx && m.role === 'user' ? {
							...m,
							state: {
								...m.state,
								...state
							},
						} : m
					)
				}
			}
		})

	}

	// set thread.state
	private _setThreadState(threadId: string, state: Partial<ThreadType['state']>, doNotRefreshMountInfo?: boolean): void {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		this._setState({
			allThreads: {
				...this.state.allThreads,
				[thread.id]: {
					...thread,
					state: {
						...thread.state,
						...state
					}
				}
			}
		}, doNotRefreshMountInfo)

	}

	private _updateThreadStateAndStore(threadId: string, state: Partial<ThreadType['state']>): void {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		const updatedThread: ThreadType = {
			...thread,
			state: {
				...thread.state,
				...state
			}
		}

		const newThreads = {
			...this.state.allThreads,
			[threadId]: updatedThread
		}

		this._storeAllThreads(newThreads)
		this._setState({ allThreads: newThreads })
	}


	// closeCurrentStagingSelectionsInThread = () => {
	// 	const currThread = this.getCurrentThreadState()

	// 	// close all stagingSelections
	// 	const closedStagingSelections = currThread.stagingSelections.map(s => ({ ...s, state: { ...s.state, isOpened: false } }))

	// 	const newThread = currThread
	// 	newThread.stagingSelections = closedStagingSelections

	// 	this.setCurrentThreadState(newThread)

	// }

	// closeCurrentStagingSelectionsInMessage: IChatThreadService['closeCurrentStagingSelectionsInMessage'] = ({ messageIdx }) => {
	// 	const currMessage = this.getCurrentMessageState(messageIdx)

	// 	// close all stagingSelections
	// 	const closedStagingSelections = currMessage.stagingSelections.map(s => ({ ...s, state: { ...s.state, isOpened: false } }))

	// 	const newMessage = currMessage
	// 	newMessage.stagingSelections = closedStagingSelections

	// 	this.setCurrentMessageState(messageIdx, newMessage)

	// }



	getCurrentThreadState = () => {
		const currentThread = this.getCurrentThread()
		return currentThread.state
	}
	setCurrentThreadState = (newState: Partial<ThreadType['state']>) => {
		this._setThreadState(this.state.currentThreadId, newState)
	}

	getAutoContinuePreference(threadId: string): boolean {
		return this.state.allThreads[threadId]?.state.autoContinueEnabled ?? false
	}

	setAutoContinuePreference(threadId: string, enabled: boolean): void {
		this._updateThreadStateAndStore(threadId, { autoContinueEnabled: enabled })
	}

	// Task planning implementation
	getTaskPlan(threadId: string): TaskPlan[] {
		return this.taskPlans[threadId] || []
	}

	// Auto-update task status based on tool execution
	private _updateTaskStatusFromToolExecution(threadId: string, toolName: string, result: 'success' | 'error'): void {
		const tasks = this.taskPlans[threadId]
		if (!tasks || tasks.length === 0) return

		// Find tasks that might relate to this tool execution
		const toolToTaskMapping: { [key: string]: string[] } = {
			'read_file': ['read', 'examine', 'analyze', 'review', 'check'],
			'edit_file': ['edit', 'modify', 'change', 'update', 'fix', 'implement'],
			'rewrite_file': ['rewrite', 'refactor', 'restructure', 'reorganize'],
			'create_file_or_folder': ['create', 'add', 'make', 'build', 'generate'],
			'delete_file_or_folder': ['delete', 'remove', 'clean', 'clear'],
			'run_command': ['run', 'execute', 'start', 'launch', 'build', 'test'],
			'search_for_files': ['search', 'find', 'locate', 'look for'],
			'ls_dir': ['list', 'explore', 'browse', 'check'],
		}

		const taskKeywords = toolToTaskMapping[toolName] || []
		if (taskKeywords.length === 0) return

		// Find pending tasks that match the tool keywords
		const matchingTasks = tasks.filter(task =>
			task.status === 'pending' || task.status === 'in_progress'
		).filter(task =>
			taskKeywords.some(keyword =>
				task.description.toLowerCase().includes(keyword)
			)
		)

		// Update the first matching task to completed (on success) or blocked (on error).
		// When multiple tasks match the same tool, only the first is updated; the
		// rest are left untouched (marking siblings pending is a separate product
		// decision, not done here).
		if (matchingTasks.length > 0) {
			const taskToUpdate = matchingTasks[0]

			if (result === 'success') {
				this.updateTaskStatus(threadId, taskToUpdate.id, 'completed')
				voidDevLog(`[chatThreadService] Auto-updated task "${taskToUpdate.description}" to completed after ${toolName} success`)
			} else {
				this.updateTaskStatus(threadId, taskToUpdate.id, 'blocked')
				voidDevLog(`[chatThreadService] Auto-updated task "${taskToUpdate.description}" to blocked after ${toolName} error`)
			}

			// NEW: Update workflow state if active
			const thread = this.state.allThreads[threadId];
			const workflow = thread?.state.activeWorkflow;

			if (workflow && workflow.status === 'active') {
				// Check if all tasks are complete
				const allTasksComplete = tasks.every(t => t.status === 'completed');
				if (allTasksComplete) {
					workflow.status = 'completed';
					voidDevLog(`[chatThreadService] Workflow "${workflow.goal}" completed`);
					this._storeAllThreads(this.state.allThreads);
					this._onDidChangeCurrentThread.fire();
				}
			}
		}
	}


	createTask(threadId: string, description: string, dependencies?: string[]): string {
		if (!this.taskPlans[threadId]) {
			this.taskPlans[threadId] = []
		}

		const task: TaskPlan = {
			id: generateUuid(),
			description,
			status: 'pending',
			dependencies,
			created_at: Date.now()
		}

		this.taskPlans[threadId].push(task)
		this._onDidChangeCurrentThread.fire() // Notify UI of change
		voidDevLog(`[chatThreadService] Created task: ${description}`)
		return task.id
	}

	updateTaskStatus(threadId: string, taskId: string, status: TaskPlan['status']): void {
		const tasks = this.taskPlans[threadId]
		if (!tasks) return

		const task = tasks.find(t => t.id === taskId)
		if (!task) return

		task.status = status
		if (status === 'completed') {
			task.completed_at = Date.now()
		}

		this._onDidChangeCurrentThread.fire() // Notify UI of change
		voidDevLog(`[chatThreadService] Updated task ${taskId} to status: ${status}`)
	}

	deleteTask(threadId: string, taskId: string): void {
		const tasks = this.taskPlans[threadId]
		if (!tasks) return

		const index = tasks.findIndex(t => t.id === taskId)
		if (index !== -1) {
			tasks.splice(index, 1)
			this._onDidChangeCurrentThread.fire() // Notify UI of change
			voidDevLog(`[chatThreadService] Deleted task ${taskId}`)
		}
	}

	clearTaskPlan(threadId: string): void {
		this.taskPlans[threadId] = []
		this._onDidChangeCurrentThread.fire() // Notify UI of change
		voidDevLog(`[chatThreadService] Cleared task plan for thread ${threadId}`)
	}

	// ==================== Student Mode Session ====================

	getStudentSession(threadId: string): StudentSession | undefined {
		return this.state.allThreads[threadId]?.state.studentSession
	}

	initStudentSession(threadId: string): StudentSession {
		const thread = this.state.allThreads[threadId]
		if (!thread) {
			throw new Error(`Thread ${threadId} not found`)
		}

		const session: StudentSession = {
			activeExercises: {},
			completedExerciseCount: 0,
			conceptsLearned: []
		}

		thread.state.studentSession = session
		this._onDidChangeCurrentThread.fire()
		return session
	}

	addExercise(threadId: string, exercise: Omit<StudentExercise, 'hintLevel' | 'status' | 'createdAt'>): StudentExercise {
		const thread = this.state.allThreads[threadId]
		if (!thread) {
			throw new Error(`Thread ${threadId} not found`)
		}

		// Initialize session if not exists
		if (!thread.state.studentSession) {
			this.initStudentSession(threadId)
		}

		const fullExercise: StudentExercise = {
			...exercise,
			hintLevel: 0,
			status: 'active',
			createdAt: Date.now()
		}

		thread.state.studentSession!.activeExercises[exercise.id] = fullExercise
		this._onDidChangeCurrentThread.fire()
		voidDevLog(`[chatThreadService] Added exercise ${exercise.id} for thread ${threadId}`)
		return fullExercise
	}

	updateExerciseHintLevel(threadId: string, exerciseId: string): number {
		const thread = this.state.allThreads[threadId]
		const exercise = thread?.state.studentSession?.activeExercises[exerciseId]

		if (!exercise) {
			voidDevWarn(`[chatThreadService] Exercise ${exerciseId} not found in thread ${threadId}`)
			return 1 // Default to level 1 if not found
		}

		// Increment hint level (max 4)
		const newLevel = Math.min(exercise.hintLevel + 1, 4)
		exercise.hintLevel = newLevel
		this._onDidChangeCurrentThread.fire()
		voidDevLog(`[chatThreadService] Updated exercise ${exerciseId} to hint level ${newLevel}`)
		return newLevel
	}

	completeExercise(threadId: string, exerciseId: string): void {
		const thread = this.state.allThreads[threadId]
		const session = thread?.state.studentSession
		const exercise = session?.activeExercises[exerciseId]

		if (!exercise || !session) {
			voidDevWarn(`[chatThreadService] Exercise ${exerciseId} not found in thread ${threadId}`)
			return
		}

		exercise.status = 'completed'
		session.completedExerciseCount++
		this._onDidChangeCurrentThread.fire()
		voidDevLog(`[chatThreadService] Completed exercise ${exerciseId}. Total completed: ${session.completedExerciseCount}`)
	}

	addConceptLearned(threadId: string, concept: string): void {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		// Initialize session if not exists
		if (!thread.state.studentSession) {
			this.initStudentSession(threadId)
		}

		const session = thread.state.studentSession!
		if (!session.conceptsLearned.includes(concept)) {
			session.conceptsLearned.push(concept)
			this._onDidChangeCurrentThread.fire()
			voidDevLog(`[chatThreadService] Added concept learned: ${concept}`)
		}
	}

	loadSkill(threadId: string, skillName: string, instructions: string): void {
		const thread = this.state.allThreads[threadId];
		if (!thread) return;

		const currentSkills = thread.state.loadedSkills || {};
		if (currentSkills[skillName]) return; // already loaded

		this._setThreadState(threadId, {
			loadedSkills: {
				...currentSkills,
				[skillName]: instructions
			}
		});
		voidDevLog(`[chatThreadService] Loaded skill: ${skillName} for thread: ${threadId}`);
	}

	// Workflow management methods
	getActiveWorkflow = (threadId: string): ThreadType['state']['activeWorkflow'] => {
		return this.state.allThreads[threadId]?.state.activeWorkflow ?? null;
	}

	setActiveWorkflowStatus = (threadId: string, status: ActiveWorkflow['status']): void => {
		const thread = this.state.allThreads[threadId];
		if (!thread?.state.activeWorkflow) return;

		thread.state.activeWorkflow.status = status;
		this._storeAllThreads(this.state.allThreads);
		this._onDidChangeCurrentThread.fire();
	}

	clearWorkflow = (threadId: string): void => {
		this._overrideWorkflow(threadId);
	}

	// ============================================
	// Composio Trigger Handling
	// ============================================

	/**
	 * Handle incoming Composio trigger events.
	 * These are received via webhook from Composio when external events occur
	 * (e.g., GitHub push, Jira ticket update, Slack message).
	 */
	handleComposioTrigger = (event: {
		triggerSlug: string;
		userId: string;
		payload: Record<string, unknown>;
		metadata: { webhookId: string; triggerId: string; timestamp: string };
	}): void => {
		voidDevLog('[ChatThreadService] Composio trigger received:', event.triggerSlug, 'webhookId:', event.metadata.webhookId);

		// Get current thread or create a new one for processing
		const threadId = this.state.currentThreadId;

		// If there's an active thread, emit an event so the UI can respond
		// The agent can then process the trigger event based on its type
		if (threadId) {
			voidDevLog('[ChatThreadService] Processing trigger event for thread:', threadId);

			// Add as a system message that the agent can process
			const thread = this.state.allThreads[threadId];
			if (thread) {
				// Emit an event that can be handled by the UI
				this._onDidChangeCurrentThread.fire();
			}
		} else {
			voidDevLog('[ChatThreadService] No active thread for trigger event');
		}

		// NOTE: A trigger queue/notification system could be added here for event-based workflows
	}

	/** Delete all messages from a given index onwards (for context menu delete/retry) */
	deleteMessagesFromIndex(threadId: string, messageIdx: number): void {
		const thread = this.state.allThreads[threadId]
		if (!thread) return
		if (messageIdx < 0 || messageIdx >= thread.messages.length) return

		const newMessages = thread.messages.slice(0, messageIdx)
		const newThreads = {
			...this.state.allThreads,
			[threadId]: {
				...thread,
				messages: newMessages,
			},
		}
		this._storeAllThreads(newThreads)
		this._setState({ allThreads: newThreads })
		this._onDidChangeCurrentThread.fire()
	}

	/** Regenerate the assistant response from a given user message */
	async retryFromMessage(threadId: string, messageIdx: number): Promise<void> {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		// Must be a user message
		const msg = thread.messages[messageIdx]
		if (!msg || msg.role !== 'user') return

		// Delete everything after this message
		this.deleteMessagesFromIndex(threadId, messageIdx + 1)

		// Re-stream from this user message
		const userMsg = msg as ChatMessage & { role: 'user' }
		await this._addUserMessageAndStreamResponse({
			userMessage: userMsg.displayContent || '',
			_chatSelections: userMsg.selections || undefined,
			threadId,
		})
	}

	/** Copy message content to clipboard */
	copyMessageContent(threadId: string, messageIdx: number): string {
		const thread = this.state.allThreads[threadId]
		if (!thread) return ''
		const msg = thread.messages[messageIdx]
		if (!msg) return ''

		if (msg.role === 'user' || msg.role === 'assistant') return msg.displayContent || ''
		if (msg.role === 'tool') {
			if (msg.type === 'success' && typeof msg.result === 'string') return msg.result
			return msg.content || ''
		}
		return ''
	}

	// ─────────────────────────────────────────────────────────────────────
	// /compact — manual context compaction
	// ─────────────────────────────────────────────────────────────────────

	/** Number of trailing ChatMessages /compact keeps verbatim. Older messages are
	 *  summarized into the snapshot. Mirrors the rolling-window `keepLastNMessages`
	 *  used by the automatic compression path. */
	private static readonly COMPACT_KEEP_LAST_N = 10

	/** Per-message char cap when rendering old messages into the summarizer prompt. */
	private static readonly COMPACT_TRANSCRIPT_PER_MSG_CAP = 1500
	/** Total char cap on the rendered transcript fed to the summarizer. */
	private static readonly COMPACT_TRANSCRIPT_TOTAL_CAP = 100_000

	clearCompaction(threadId: string): void {
		const thread = this.state.allThreads[threadId]
		if (!thread) return
		if (!thread.state.compaction) return
		this._updateThreadStateAndStore(threadId, { compaction: undefined })
		this._notificationService.notify({
			severity: Severity.Info,
			message: 'Compaction cleared. The full conversation history is now sent to the model again.',
		})
	}

	async compactThread(threadId: string, focusInstructions?: string): Promise<void> {
		const thread = this.state.allThreads[threadId]
		if (!thread) return

		const chatMessages = thread.messages ?? []
		const keepN = Math.min(ChatThreadService.COMPACT_KEEP_LAST_N, chatMessages.length)
		const compactedCount = Math.max(0, chatMessages.length - keepN)

		if (compactedCount === 0) {
			this._notificationService.notify({
				severity: Severity.Info,
				message: 'Nothing to compact yet — the conversation is short enough to send in full.',
			})
			return
		}

		// Stop any running stream first — compaction rewrites the LLM-facing context,
		// so we don't want a turn in flight to race against the snapshot change.
		await this.abortRunning(threadId)

		const oldMessages = chatMessages.slice(0, compactedCount)

		// Fire PreCompact *before* summarizing so hooks can inject preservation context
		// (Claude Code semantics). The hook's additionalContext is folded into the
		// summarizer prompt so plugins can bias what gets retained. Non-blocking.
		let hookContext = ''
		try {
			const hookRes = await this._hookService.firePreCompact(threadId, oldMessages)
			if (hookRes.additionalContext) hookContext = hookRes.additionalContext
		} catch (err) {
			voidDevWarn('[compact] PreCompact hook threw (non-blocking):', err)
		}

		const focus = focusInstructions?.trim() || undefined
		let summaryText = ''
		try {
			summaryText = await this._summarizeForCompaction(oldMessages, focus, hookContext)
		} catch (err) {
			voidDevWarn('[compact] LLM summarizer failed, falling back to heuristic summary:', err)
		}
		if (!summaryText.trim()) {
			summaryText = this._heuristicSummary(oldMessages)
		}

		const snapshot: CompactionSnapshot = {
			summaryText,
			compactedChatMessageCount: compactedCount,
			keptChatMessageCount: keepN,
			createdAt: new Date().toISOString(),
			trigger: 'manual',
			focusInstructions: focus,
		}
		this._updateThreadStateAndStore(threadId, { compaction: snapshot })

		this._notificationService.notify({
			severity: Severity.Info,
			message: `Conversation compacted: ${compactedCount} message${compactedCount === 1 ? '' : 's'} → summary${focus ? ` (focus: ${focus})` : ''}. ${keepN} recent message${keepN === 1 ? '' : 's'} kept. The summary is now used for context; full history is retained for rewind. Use /compact clear to undo.`,
		})
	}

	/**
	 * Render the given ChatMessages into a readable transcript for the summarizer,
	 * skipping checkpoints/interrupted tools. Each message is capped, and the whole
	 * transcript is capped to keep the summarizer prompt bounded.
	 */
	private _renderChatMessagesToTranscript(messages: ChatMessage[]): string {
		const lines: string[] = []
		let total = 0
		const perMsgCap = ChatThreadService.COMPACT_TRANSCRIPT_PER_MSG_CAP
		const totalCap = ChatThreadService.COMPACT_TRANSCRIPT_TOTAL_CAP
		for (const m of messages) {
			if (m.role === 'checkpoint' || m.role === 'interrupted_streaming_tool') continue
			let body = ''
			if (m.role === 'user') body = m.displayContent || m.content || ''
			else if (m.role === 'assistant') body = m.displayContent || ''
			else if (m.role === 'tool') {
				const label = m.name || 'tool'
				let result = ''
				if (m.type === 'success' && typeof m.result === 'string') result = m.result
				else result = m.content || ''
				body = `[tool result: ${label}] ${result}`
			}
			if (!body) continue
			if (body.length > perMsgCap) body = body.slice(0, perMsgCap) + '\n…[truncated]'
			const line = `[${m.role}] ${body}`
			if (total + line.length > totalCap) break
			lines.push(line)
			total += line.length
		}
		return lines.join('\n\n')
	}

	/**
	 * Ask the model to summarize the compacted region via a one-shot subagent call
	 * (the same `runSubagentSync` path the prompt-type hooks use). Returns '' on any
	 * failure so the caller can fall back to the heuristic summary.
	 */
	private async _summarizeForCompaction(oldMessages: ChatMessage[], focusInstructions: string | undefined, hookContext: string): Promise<string> {
		const transcript = this._renderChatMessagesToTranscript(oldMessages)
		if (!transcript.trim()) return ''

		const focusLine = focusInstructions
			? `\n\nThe user wants the summary to FOCUS ON: ${focusInstructions}. Emphasize details related to this focus.`
			: ''
		const preserveLine = `\n\nWhen summarizing, always preserve:\n- The current task objective and any acceptance criteria\n- File paths that were read or modified\n- Tool results and error messages\n- Decisions made and the reasoning behind them`
		const hookLine = hookContext ? `\n\nAdditional context to preserve (from a PreCompact hook):\n${hookContext}` : ''

		const prompt = `You are a conversation summarizer. Summarize the following earlier portion of a coding conversation into a concise but complete summary that a developer's AI assistant can use as background context to continue the work. Do NOT use any tools — respond with only the summary prose.${focusLine}${preserveLine}${hookLine}\n\n--- BEGIN EARLIER CONVERSATION ---\n${transcript}\n--- END EARLIER CONVERSATION ---\n\nWrite the summary now. Do not add headings or preface; just the summary.`

		const result = await this._subagentService.runSubagentSync({
			parentThreadId: null,
			description: 'Summarize earlier conversation for /compact',
			subagentType: 'general',
			prompt,
			// Give a trivial tool so the run is well-formed; the prompt instructs the
			// model not to use it. (Same approach the prompt-type hooks use.)
			tools: ['read_file'],
			background: false,
			title: 'compact:summarizer',
		})
		return (result.fullText || '').trim()
	}

	/** Heuristic extractive fallback when the LLM summarizer is unavailable or fails.
	 *  Produces a short role-tagged preview per old message, similar in spirit to the
	 *  ContextCompressionService summary but operating on ChatMessages. */
	private _heuristicSummary(oldMessages: ChatMessage[]): string {
		const parts: string[] = []
		for (const m of oldMessages) {
			if (m.role === 'checkpoint' || m.role === 'interrupted_streaming_tool') continue
			let body = ''
			if (m.role === 'user') body = m.displayContent || m.content || ''
			else if (m.role === 'assistant') body = m.displayContent || ''
			else if (m.role === 'tool') body = m.content || ''
			if (!body) continue
			const preview = body.split(/\n\n|\n/).slice(0, 2).join(' ').slice(0, 400)
			if (preview.length > 10) parts.push(`[${m.role}]: ${preview}`)
		}
		const body = parts.length > 0
			? parts.join('\n')
			: '(No extractable content from the compacted region.)'
		return `[PREVIOUS CONVERSATION SUMMARY - ${oldMessages.length} messages condensed. This is background context from earlier in the conversation, NOT a new request from the user. Do not respond to it directly.]\n\n${body}\n\n[End of summary]`
	}

	/**
	 * Auto-compact a thread when it grows past the configured threshold. Called from
	 * the run-agent wrapper after a turn completes and the thread is idle.
	 */
	private async _maybeAutoCompact(threadId: string): Promise<void> {
		const { globalSettings } = this._settingsService.state
		if (!globalSettings.enableAutoCompact) return

		const thread = this.state.allThreads[threadId]
		if (!thread) return
		if (this.streamState[threadId]?.isRunning !== undefined && this.streamState[threadId]?.isRunning !== 'idle') return

		const chatMessages = thread.messages ?? []
		const threshold = Math.max(10, globalSettings.autoCompactThresholdMessages ?? 30)
		const minSinceLast = Math.max(5, globalSettings.autoCompactMinMessagesSinceLast ?? 20)

		const existing = thread.state.compaction
		const messagesSinceLast = existing ? Math.max(0, chatMessages.length - existing.compactedChatMessageCount) : chatMessages.length
		const shouldCompact = chatMessages.length >= threshold && messagesSinceLast >= minSinceLast
		if (!shouldCompact) return

		// Avoid auto-compacting if there is an active workflow/plan in progress.
		if (thread.state.activeWorkflow?.status === 'active') return

		const keepN = Math.min(ChatThreadService.COMPACT_KEEP_LAST_N, chatMessages.length)
		const compactedCount = Math.max(0, chatMessages.length - keepN)
		if (compactedCount === 0) return

		const oldMessages = chatMessages.slice(0, compactedCount)
		let hookContext = ''
		try {
			const hookRes = await this._hookService.firePreCompact(threadId, oldMessages)
			if (hookRes.additionalContext) hookContext = hookRes.additionalContext
		} catch (err) {
			voidDevWarn('[auto-compact] PreCompact hook threw (non-blocking):', err)
		}

		let summaryText = ''
		try {
			summaryText = await this._summarizeForCompaction(oldMessages, undefined, hookContext)
		} catch (err) {
			voidDevWarn('[auto-compact] LLM summarizer failed, falling back to heuristic summary:', err)
		}
		if (!summaryText.trim()) {
			summaryText = this._heuristicSummary(oldMessages)
		}

		const snapshot: CompactionSnapshot = {
			summaryText,
			compactedChatMessageCount: compactedCount,
			keptChatMessageCount: keepN,
			createdAt: new Date().toISOString(),
			trigger: 'auto',
		}
		this._updateThreadStateAndStore(threadId, { compaction: snapshot })

		// Surface to the UI with the same toast used by rolling-window compression.
		triggerCompressionNotification({
			originalMessageCount: chatMessages.length,
			finalMessageCount: keepN + 1, // kept messages + summary
			originalTokens: chatMessages.length * 200, // rough estimate; we don't token-count here
			finalTokens: (keepN + 1) * 200,
			compressionRatio: Math.round(((keepN + 1) / Math.max(1, chatMessages.length)) * 100),
			messagesRemoved: compactedCount,
			messagesSummarized: compactedCount,
		}, threadId)
	}

	// gets `staging` and `setStaging` of the currently focused element, given the index of the currently selected message (or undefined if no message is selected)

	getCurrentMessageState(messageIdx: number): UserMessageState {
		const currMessage = this.getCurrentThread()?.messages?.[messageIdx]
		if (!currMessage || currMessage.role !== 'user') return defaultMessageState
		return currMessage.state
	}
	setCurrentMessageState(messageIdx: number, newState: Partial<UserMessageState>) {
		const currMessage = this.getCurrentThread()?.messages?.[messageIdx]
		if (!currMessage || currMessage.role !== 'user') return
		this._setCurrentMessageState(newState, messageIdx)
	}



}

registerSingleton(IChatThreadService, ChatThreadService, InstantiationType.Delayed);
