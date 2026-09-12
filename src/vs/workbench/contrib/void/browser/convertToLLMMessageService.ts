import { Disposable } from '../../../../base/common/lifecycle.js';
import { IModelService } from '../../../../editor/common/services/model.js';
import { registerSingleton, InstantiationType } from '../../../../platform/instantiation/common/extensions.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { IWorkspaceContextService } from '../../../../platform/workspace/common/workspace.js';
import { IEditorService } from '../../../services/editor/common/editorService.js';
import { ChatMessage, CompactionSnapshot } from '../common/chatThreadServiceTypes.js';
import { getIsReasoningEnabledState, getReservedOutputTokenSpace, getModelCapabilities } from '../common/modelCapabilities.js';
import { chat_systemMessage, InternalToolInfo } from '../common/prompt/prompts.js';
import { AnthropicLLMChatMessage, AnthropicReasoning, GeminiLLMChatMessage, LLMChatMessage, LLMFIMMessage, OpenAILLMChatMessage, RawToolParamsObj } from '../common/sendLLMMessageTypes.js';
import { IVoidSettingsService } from '../common/voidSettingsService.js';
import { ChatMode, FeatureName, ModelSelection, ProviderName } from '../common/voidSettingsTypes.js';
import { IDirectoryStrService } from '../common/directoryStrService.js';
import { ITerminalToolService } from './terminalToolService.js';
import { IVoidModelService } from '../common/voidModelService.js';
import { URI } from '../../../../base/common/uri.js';
import { EndOfLinePreference } from '../../../../editor/common/model.js';
import { ToolName } from '../common/toolsServiceTypes.js';
import { IMCPService } from '../common/mcpService.js';
import { IACPService } from '../common/acpService.js';
import { IComposioService } from '../common/composioService.js';
import { ITokenCountingService } from '../common/tokenCountingService.js';
import { ContextCompressionService } from '../common/contextCompressionService.js';

export const EMPTY_MESSAGE = '(empty message)'



export type SimpleLLMMessage = {
	role: 'tool';
	content: string;
	id: string;
	name: ToolName;
	rawParams: RawToolParamsObj;
	thought_signature?: string;
} | {
	role: 'user';
	content: string;
} | {
	role: 'assistant';
	content: string;
	reasoning?: string;
	anthropicReasoning: AnthropicReasoning[] | null;
}

/**
 * A real prompt-usage measurement from the last successful LLM call, used to
 * anchor context-size accounting (ported from a-coder-cli's compaction design):
 * instead of re-tokenizing the entire thread on every message, we take the
 * measured `promptTokens` as ground truth for the covered prefix and only
 * cheaply estimate the trailing messages added since.
 */
export type UsageAnchor = {
	/** Measured prompt tokens (system message + converted thread messages). */
	promptTokens: number
	/** Number of thread ChatMessages the measurement covered. */
	coveredThreadMessageCount: number
	/** Compaction snapshot count at measurement time — anchor is invalid if this changed. */
	compactedChatMessageCountAtAnchor: number
	providerName: string
	modelName: string
}



const CHARS_PER_TOKEN = 4 // assume abysmal chars per token
const TRIM_TO_LEN = 120




// convert messages as if about to send to openai
/*
reference - https://platform.openai.com/docs/guides/function-calling#function-calling-steps
openai MESSAGE (role=assistant):
"tool_calls":[{
	"type": "function",
	"id": "call_12345xyz",
	"function": {
	"name": "get_weather",
	"arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
}]

openai RESPONSE (role=user):
{   "role": "tool",
	"tool_call_id": tool_call.id,
	"content": str(result)    }

also see
openai on prompting - https://platform.openai.com/docs/guides/reasoning#advice-on-prompting
openai on developer system message - https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command
*/


// convert messages as if about to send to anthropic
/*
https://docs.anthropic.com/en/docs/build-with-claude/tool-use#tool-use-examples
anthropic MESSAGE (role=assistant):
"content": [{
	"type": "text",
	"text": "<thinking>I need to call the get_weather function, and the user wants SF, which is likely San Francisco, CA.</thinking>"
}, {
	"type": "tool_use",
	"id": "toolu_01A09q90qw90lq917835lq9",
	"name": "get_weather",
	"input": { "location": "San Francisco, CA", "unit": "celsius" }
}]
anthropic RESPONSE (role=user):
"content": [{
	"type": "tool_result",
	"tool_use_id": "toolu_01A09q90qw90lq917835lq9",
	"content": "15 degrees"
}]


Converts:
assistant: ...content
tool: (id, name, params)
->
assistant: ...content, call(name, id, params)
user: ...content, result(id, content)
*/

type AnthropicOrOpenAILLMMessage = AnthropicLLMChatMessage | OpenAILLMChatMessage

// Convert SimpleLLMMessage[] to OpenAI format with proper tool_calls and tool_call_id
const prepareMessages_openai_tools = (messages: SimpleLLMMessage[]): OpenAILLMChatMessage[] => {
	const newMessages: OpenAILLMChatMessage[] = []

	for (let i = 0; i < messages.length; i += 1) {
		const currMsg = messages[i]

		if (currMsg.role === 'assistant') {
			// Find all consecutive tool messages following this assistant message
			const toolCalls: any[] = []
			let j = i + 1
			while (j < messages.length && messages[j].role === 'tool') {
				const toolMsg = messages[j]
				if (toolMsg.role === 'tool') {
					toolCalls.push({
						type: 'function',
						id: toolMsg.id,
						function: {
							name: toolMsg.name,
							arguments: JSON.stringify(toolMsg.rawParams),
							thought_signature: toolMsg.thought_signature,
						}
					})
				}
				j++
			}

			// Get signature and reasoning from anthropicReasoning if available
			const signature = currMsg.anthropicReasoning?.[0]?.type === 'thinking' ? currMsg.anthropicReasoning[0].signature : undefined

			const assistantMsg: any = {
				role: 'assistant',
				content: currMsg.content || (toolCalls.length > 0 ? null : ''),
				reasoning: currMsg.reasoning,
				thought_signature: signature,
			}

			if (toolCalls.length > 0) {
				assistantMsg.tool_calls = toolCalls
				// If there are tool calls, the thought_signature might come from the first tool call
				const firstToolMsg = messages[i + 1]
				if (!assistantMsg.thought_signature && firstToolMsg.role === 'tool' && firstToolMsg.thought_signature) {
					assistantMsg.thought_signature = firstToolMsg.thought_signature
				}
			}

			newMessages.push(assistantMsg)
			continue
		}

		if (currMsg.role === 'user') {
			newMessages.push({
				role: 'user',
				content: currMsg.content,
			})
			continue
		}

		if (currMsg.role === 'tool') {
			// Convert to OpenAI tool format with tool_call_id
			// We also include 'name' which is helpful for some proxies that translate to Gemini
			newMessages.push({
				role: 'tool',
				tool_call_id: currMsg.id,
				name: currMsg.name,
				content: currMsg.content,
			})
			continue
		}
	}

	return newMessages
}

const prepareMessages_anthropic_tools = (messages: SimpleLLMMessage[], supportsAnthropicReasoning: boolean): AnthropicOrOpenAILLMMessage[] => {
	const newMessages: (AnthropicLLMChatMessage | (SimpleLLMMessage & { role: 'tool' }))[] = messages;

	let lastAssistantIdx = -1

	for (let i = 0; i < messages.length; i += 1) {
		const currMsg = messages[i]

		// add anthropic reasoning
		if (currMsg.role === 'assistant') {
			lastAssistantIdx = i
			if (currMsg.anthropicReasoning && supportsAnthropicReasoning) {
				const content = currMsg.content
				newMessages[i] = {
					role: 'assistant',
					content: content ? [...currMsg.anthropicReasoning, { type: 'text' as const, text: content }] : currMsg.anthropicReasoning
				}
			}
			else {
				newMessages[i] = {
					role: 'assistant',
					content: currMsg.content,
					// strip away anthropicReasoning
				}
			}
			continue
		}

		if (currMsg.role === 'user') {
			newMessages[i] = {
				role: 'user',
				content: currMsg.content,
			}
			continue
		}

		if (currMsg.role === 'tool') {
			// add anthropic tools
			const prevMsg = lastAssistantIdx !== -1 ? newMessages[lastAssistantIdx] : undefined

			// make it so the assistant called the tool
			if (prevMsg?.role === 'assistant') {
				if (typeof prevMsg.content === 'string') prevMsg.content = [{ type: 'text', text: prevMsg.content }]
				prevMsg.content.push({ type: 'tool_use', id: currMsg.id, name: currMsg.name, input: currMsg.rawParams, signature: currMsg.thought_signature })
			}

			// turn each tool into a user message with tool results at the end
			newMessages[i] = {
				role: 'user',
				content: [{ type: 'tool_result', tool_use_id: currMsg.id, content: currMsg.content, name: currMsg.name }]
			}
			continue
		}

	}

	// we just removed the tools
	return newMessages as AnthropicLLMChatMessage[]
}


type GeminiUserPart = (GeminiLLMChatMessage & { role: 'user' })['parts'][0]
type GeminiModelPart = (GeminiLLMChatMessage & { role: 'model' })['parts'][0]

const prepareGeminiMessages = (messages: AnthropicLLMChatMessage[]) => {
	const toolIdToName = new Map<string, string>()
	const messages2: GeminiLLMChatMessage[] = messages.map((m): GeminiLLMChatMessage | null => {
		if (m.role === 'assistant') {
			if (typeof m.content === 'string') {
				return { role: 'model', parts: [{ text: m.content }] }
			}
			else {
				const parts: GeminiModelPart[] = m.content.map((c): GeminiModelPart | null => {
					if (c.type === 'text') {
						return { text: c.text }
					}
					else if (c.type === 'thinking') {
						return { text: c.thinking, thought: true }
					}
					else if (c.type === 'tool_use') {
						if (c.id && c.name) toolIdToName.set(c.id, c.name)
						return { functionCall: { id: c.id, name: c.name, args: c.input, thought_signature: c.signature } }
					}
					else return null
				}).filter(m => !!m) as GeminiModelPart[]
				return { role: 'model', parts, }
			}
		}
		else if (m.role === 'user') {
			if (typeof m.content === 'string') {
				return { role: 'user', parts: [{ text: m.content }] } satisfies GeminiLLMChatMessage
			}
			else {
				const parts: GeminiUserPart[] = m.content.map((c): GeminiUserPart | null => {
					if (c.type === 'text') {
						return { text: c.text }
					}
					else if (c.type === 'tool_result') {
						const name = c.name || toolIdToName.get(c.tool_use_id)
						if (!name) return null
						return { functionResponse: { id: c.tool_use_id, name: name as ToolName, response: { output: c.content } } }
					}
					else return null
				}).filter(m => !!m)
				return { role: 'user', parts, }
			}

		}
		else return null
	}).filter(m => !!m)

	return messages2
}

// Convert messages for models using marker tokens (<|tool_call_start|>, etc.)
const prepareMessages_marker_tools = (messages: SimpleLLMMessage[]): OpenAILLMChatMessage[] => {
	const newMessages: OpenAILLMChatMessage[] = []

	// Helper to append text to the last message if it's the right role
	const appendToLast = (role: string, text: string) => {
		const last = newMessages[newMessages.length - 1]
		if (last && last.role === role && typeof last.content === 'string') {
			last.content += '\n' + text
			return true
		}
		return false
	}

	for (let i = 0; i < messages.length; i += 1) {
		const currMsg = messages[i]

		if (currMsg.role === 'assistant') {
			newMessages.push({
				role: 'assistant',
				content: currMsg.content || '',
			})
			continue
		}

		if (currMsg.role === 'user') {
			newMessages.push({
				role: 'user',
				content: currMsg.content,
			})
			continue
		}

		if (currMsg.role === 'tool') {
			// 1. Append tool call to PREVIOUS assistant message
			// <|tool_call_start|>{"name": "...", "arguments": {...}}<|tool_call_end|>
			const callStr = `<|tool_call_start|>${JSON.stringify({ name: currMsg.name, arguments: currMsg.rawParams })}<|tool_call_end|>`
			
			// Try to append to last assistant message
			if (!appendToLast('assistant', callStr)) {
				// If no previous assistant message, insert a fake one (should rarely happen in valid flows)
				newMessages.push({ role: 'assistant', content: callStr })
			}

			// 2. Add tool result as USER message (or append if multiple tools)
			// <|tool_response_start|>{"name": "...", "content": "..."}<|tool_response_end|>
			const resultStr = `<|tool_response_start|>${JSON.stringify({ name: currMsg.name, content: currMsg.content })}<|tool_response_end|>`
			
			// We typically want tool results to be distinct messages or grouped.
			// For markers, they are usually just text blocks in the conversation.
			newMessages.push({
				role: 'user',
				content: resultStr,
			})
			continue
		}
	}

	return newMessages
}


// --- CHAT ---

const prepareOpenAIOrAnthropicMessages = ({
	messages: messages_,
	systemMessage,
	aiInstructions,
	supportsSystemMessage,
	specialToolFormat,
	supportsAnthropicReasoning,
	contextWindow,
	reservedOutputTokenSpace,
	providerName,
}: {
	messages: SimpleLLMMessage[],
	systemMessage: string,
	aiInstructions: string,
	supportsSystemMessage: false | 'system-role' | 'developer-role' | 'separated',
	specialToolFormat: 'openai-style' | 'anthropic-style' | 'gemini-style' | 'marker-style',
	supportsAnthropicReasoning: boolean,
	contextWindow: number,
	reservedOutputTokenSpace: number | null | undefined,
	providerName: ProviderName
}): { messages: AnthropicOrOpenAILLMMessage[], separateSystemMessage: string | undefined } => {

	reservedOutputTokenSpace = Math.max(
		contextWindow * 1 / 2, // reserve at least 1/4 of the token window length
		reservedOutputTokenSpace ?? 4_096 // defaults to 4096
	)
	// PERFORMANCE: Use shallow copy instead of deepClone - we create new objects when modifying anyway
	let messages: (SimpleLLMMessage | { role: 'system', content: string })[] = [...messages_]

	// ================ system message ================
	// Include system message in weight calculations for context trimming

	const sysMsgParts: string[] = []
	if (aiInstructions) sysMsgParts.push(`GUIDELINES (from the user's .a-coder-rules file):\n${aiInstructions}`)
	if (systemMessage) sysMsgParts.push(systemMessage)
	const combinedSystemMessage = sysMsgParts.join('\n\n')

	messages.unshift({ role: 'system', content: combinedSystemMessage })

	// ================ trim ================
	messages = messages.map(m => ({ ...m, content: m.role !== 'tool' ? m.content.trim() : m.content }))

	type MesType = (typeof messages)[0]

	// ================ fit into context ================

	// the higher the weight, the higher the desire to truncate - TRIM HIGHEST WEIGHT MESSAGES
	const alreadyTrimmedIdxes = new Set<number>()
	const weight = (message: MesType, messages: MesType[], idx: number) => {
		const base = message.content.length

		let multiplier: number
		multiplier = 1 + (messages.length - 1 - idx) / messages.length // slow rampdown from 2 to 1 as index increases
		if (message.role === 'user') {
			multiplier *= 1
		}
		else if (message.role === 'system') {
			multiplier *= .01 // very low weight
		}
		else if (message.role === 'tool') {
			multiplier *= .02 // very low weight - tool results are critical for LLM function
		}
		else {
			multiplier *= 10 // llm tokens are far less valuable than user tokens
		}

		// any already modified message should not be trimmed again
		if (alreadyTrimmedIdxes.has(idx)) {
			multiplier = 0
		}
		// 1st and last messages should be very low weight
		if (idx <= 1 || idx >= messages.length - 1 - 3) {
			multiplier *= .05
		}
		return base * multiplier
	}

	const _findLargestByWeight = (messages_: MesType[]) => {
		let largestIndex = -1
		let largestWeight = -Infinity
		for (let i = 0; i < messages.length; i += 1) {
			const m = messages[i]
			const w = weight(m, messages_, i)
			if (w > largestWeight) {
				largestWeight = w
				largestIndex = i
			}
		}
		return largestIndex
	}

	let totalLen = 0
	for (const m of messages) { totalLen += m.content.length }
	const charsNeedToTrim = totalLen - Math.max(
		(contextWindow - reservedOutputTokenSpace) * CHARS_PER_TOKEN, // can be 0, in which case charsNeedToTrim=everything, bad
		5_000 // ensure we don't trim at least 5k chars (just a random small value)
	)


	// <----------------------------------------->
	// 0                      |    |             |
	//                        |    contextWindow |
	//                     contextWindow - maxOut|putTokens
	//                                          totalLen
	let remainingCharsToTrim = charsNeedToTrim
	let i = 0

	while (remainingCharsToTrim > 0) {
		i += 1
		if (i > 100) break

		const trimIdx = _findLargestByWeight(messages)
		const m = messages[trimIdx]

		// if can finish here, do
		const numCharsWillTrim = m.content.length - TRIM_TO_LEN
		if (numCharsWillTrim > remainingCharsToTrim) {
			// trim remainingCharsToTrim + '...'.length chars
			m.content = m.content.slice(0, m.content.length - remainingCharsToTrim - '...'.length).trim() + '...'
			break
		}

		remainingCharsToTrim -= numCharsWillTrim
		m.content = m.content.substring(0, TRIM_TO_LEN - '...'.length) + '...'
		alreadyTrimmedIdxes.add(trimIdx)
	}

	// ================ system message hack ================
	const newSysMsg = messages.shift()!.content


	// ================ tools and anthropicReasoning ================
	// Extract system message after trimming; SimpleLLMMessage[] is now valid without system role

	let llmChatMessages: AnthropicOrOpenAILLMMessage[] = []
	if (specialToolFormat === 'anthropic-style') {
		llmChatMessages = prepareMessages_anthropic_tools(messages as SimpleLLMMessage[], supportsAnthropicReasoning)
	}
	else if (specialToolFormat === 'openai-style') {
		// Convert to proper OpenAI format with tool_calls and tool_call_id
		llmChatMessages = prepareMessages_openai_tools(messages as SimpleLLMMessage[])
	}
	else if (specialToolFormat === 'marker-style') {
		llmChatMessages = prepareMessages_marker_tools(messages as SimpleLLMMessage[])
	}
	else {
		throw new Error(`Model from provider "${providerName}" does not support native tool calling.`)
	}
	const llmMessages = llmChatMessages


	// ================ system message add as first llmMessage ================

	let separateSystemMessageStr: string | undefined = undefined

	// if supports system message
	if (supportsSystemMessage) {
		if (supportsSystemMessage === 'separated')
			separateSystemMessageStr = newSysMsg
		else if (supportsSystemMessage === 'system-role')
			llmMessages.unshift({ role: 'system', content: newSysMsg }) // add new first message
		else if (supportsSystemMessage === 'developer-role')
			llmMessages.unshift({ role: 'developer', content: newSysMsg }) // add new first message
	}
	// if does not support system message
	else {
		const newFirstMessage = {
			role: 'user',
			content: `<SYSTEM_MESSAGE>\n${newSysMsg}\n</SYSTEM_MESSAGE>\n${llmMessages[0].content}`
		} as const
		llmMessages.splice(0, 1) // delete first message
		llmMessages.unshift(newFirstMessage) // add new first message
	}


	// ================ no empty message ================
	for (let i = 0; i < llmMessages.length; i += 1) {
		const currMsg: AnthropicOrOpenAILLMMessage = llmMessages[i]
		const nextMsg: AnthropicOrOpenAILLMMessage | undefined = llmMessages[i + 1]

		if (currMsg.role === 'tool') continue

		// if content is a string, replace string with empty msg
		if (typeof currMsg.content === 'string') {
			currMsg.content = currMsg.content || EMPTY_MESSAGE
		}
		else if (currMsg.content === null) {
			// already null, leave it as is (happens with tool_calls)
			continue
		}
		else {
			// allowed to be empty if has a tool in it or following it
			if (currMsg.content.find(c => c.type === 'tool_result' || c.type === 'tool_use')) {
				currMsg.content = currMsg.content.filter(c => !(c.type === 'text' && !c.text)) as any
				continue
			}
			if (nextMsg?.role === 'tool') continue

			// replace any empty text entries with empty msg, and make sure there's at least 1 entry
			for (const c of currMsg.content) {
				if (c.type === 'text') c.text = c.text || EMPTY_MESSAGE
			}
			if (currMsg.content.length === 0) currMsg.content = [{ type: 'text', text: EMPTY_MESSAGE }]
		}
	}

	return {
		messages: llmMessages,
		separateSystemMessage: separateSystemMessageStr,
	} as const
}




const prepareMessages = (params: {
	messages: SimpleLLMMessage[],
	systemMessage: string,
	aiInstructions: string,
	supportsSystemMessage: false | 'system-role' | 'developer-role' | 'separated',
	specialToolFormat: 'openai-style' | 'anthropic-style' | 'gemini-style' | 'marker-style',
	supportsAnthropicReasoning: boolean,
	contextWindow: number,
	reservedOutputTokenSpace: number | null | undefined,
	providerName: ProviderName
}): { messages: LLMChatMessage[], separateSystemMessage: string | undefined } => {

	const specialFormat = params.specialToolFormat // this is just for ts stupidness

	// if need to convert to gemini style of messaes, do that (treat as anthropic style, then convert to gemini style)
	if (params.providerName === 'gemini' || specialFormat === 'gemini-style') {
		const res = prepareOpenAIOrAnthropicMessages({ ...params, specialToolFormat: 'anthropic-style' })
		const messages = res.messages as AnthropicLLMChatMessage[]
		const messages2 = prepareGeminiMessages(messages)
		return { messages: messages2, separateSystemMessage: res.separateSystemMessage }
	}

	return prepareOpenAIOrAnthropicMessages({ ...params, specialToolFormat: specialFormat, providerName: params.providerName })
}




export interface IConvertToLLMMessageService {
	readonly _serviceBrand: undefined;
	prepareLLMSimpleMessages: (opts: { simpleMessages: SimpleLLMMessage[], systemMessage: string, modelSelection: ModelSelection | null, featureName: FeatureName }) => { messages: LLMChatMessage[], separateSystemMessage: string | undefined }
	prepareLLMChatMessages: (opts: { chatMessages: ChatMessage[], chatMode: ChatMode, modelSelection: ModelSelection | null, loadedSkills?: { [name: string]: string }, orchestrationResult?: { suggestions: Array<{ toolName: string; toolParams?: Record<string, unknown>; reasoning: string; confidence: 'high' | 'medium' | 'low'; }>; reasoning: string; summary: string; }, compaction?: CompactionSnapshot, usageAnchor?: UsageAnchor }) => Promise<{ messages: LLMChatMessage[], separateSystemMessage: string | undefined, tokenUsage: { used: number, total: number, percentage: number }, compressionStats?: { originalMessageCount: number, finalMessageCount: number, originalTokens: number, finalTokens: number, compressionRatio: number, messagesRemoved: number, messagesSummarized: number } }>
	prepareFIMMessage(opts: { messages: LLMFIMMessage, }): { prefix: string, suffix: string, stopTokens: string[] }
	updateTokenRatio(modelName: string, estimatedTokens: number, actualTokens: number): void
	// Builds the system message for a subagent run: the subagent's role prompt
	// prepended to the full chat system message (workspace context + tool
	// descriptions), with the tool set restricted to `allowedTools`. The tool
	// restriction applies to text tool descriptions (xml/marker models); native
	// tool defs are restricted separately by the LLM transport via the
	// `allowedTools` param on sendLLMMessage.
	buildSubagentSystemMessage(opts: { rolePrompt: string, modelSelection: ModelSelection | null, allowedTools: string[] | undefined, allowExternalTools?: boolean }): Promise<string>
}

export const IConvertToLLMMessageService = createDecorator<IConvertToLLMMessageService>('ConvertToLLMMessageService');


class ConvertToLLMMessageService extends Disposable implements IConvertToLLMMessageService {
	_serviceBrand: undefined;

	private readonly tokenCountingService: ITokenCountingService;
	private readonly compressionService: ContextCompressionService;

	constructor(
		@IModelService private readonly modelService: IModelService,
		@IWorkspaceContextService private readonly workspaceContextService: IWorkspaceContextService,
		@IEditorService private readonly editorService: IEditorService,
		@IDirectoryStrService private readonly directoryStrService: IDirectoryStrService,
		@ITerminalToolService private readonly terminalToolService: ITerminalToolService,
		@IVoidSettingsService private readonly voidSettingsService: IVoidSettingsService,
		@IVoidModelService private readonly voidModelService: IVoidModelService,
		@IMCPService private readonly mcpService: IMCPService,
		@IACPService private readonly acpService: IACPService,
		@IComposioService private readonly composioService: IComposioService,
		@ITokenCountingService tokenCountingService: ITokenCountingService,
	) {
		super();

		// Initialize token counting and compression services
		this.tokenCountingService = tokenCountingService;
		this.compressionService = new ContextCompressionService(this.tokenCountingService);
	}

	// Read .a-coder-rules files from workspace folders
	private _getVoidRulesFileContents(): string {
		try {
			const workspaceFolders = this.workspaceContextService.getWorkspace().folders;
			let voidRules = '';
			for (const folder of workspaceFolders) {
				const uri = URI.joinPath(folder.uri, '.a-coder-rules')
				const { model } = this.voidModelService.getModel(uri)
				if (!model) continue
				voidRules += model.getValue(EndOfLinePreference.LF) + '\n\n';
			}
			return voidRules.trim();
		}
		catch (e) {
			return ''
		}
	}

	// Get combined AI instructions from settings and .a-coder-rules files
	private _getCombinedAIInstructions(): string {
		const globalAIInstructions = this.voidSettingsService.state.globalSettings.aiInstructions;
		const voidRulesFileContent = this._getVoidRulesFileContents();

		const ans: string[] = []
		if (globalAIInstructions) ans.push(globalAIInstructions)
		if (voidRulesFileContent) ans.push(voidRulesFileContent)
		return ans.join('\n\n')
	}


	// system message
	private _generateChatMessagesSystemMessage = async (chatMode: ChatMode, specialToolFormat: 'openai-style' | 'anthropic-style' | 'gemini-style' | 'marker-style' | undefined, modelSelection: ModelSelection | null, allowedTools?: string[], allowExternalTools?: boolean) => {
		const workspaceFolders = this.workspaceContextService.getWorkspace().folders.map(f => f.uri.fsPath)

		const openedURIs = this.modelService.getModels().filter(m => m.isAttachedToEditor()).map(m => m.uri.fsPath) || [];
		const activeURI = this.editorService.activeEditor?.resource?.fsPath;

		const directoryStr = await this.directoryStrService.getAllDirectoriesStr({
			cutOffMessage: chatMode === 'code' || chatMode === 'plan' ?
				`...Directories string cut off, use tools to read more...`
				: `...Directories string cut off, ask user for more if necessary...`
		})

		const mcpTools = this.mcpService.getMCPTools()

		// Get ACP agents if configured
		const acpTools = this.acpService.getACPAgents()

		// Get Composio meta tools if configured
		const composioTools = await this._getComposioTools()

		const persistentTerminalIDs = this.terminalToolService.listPersistentTerminalIds()

		// Get student level for student mode
		const studentLevel = chatMode === 'learn' ? this.voidSettingsService.state.globalSettings.studentLevel : undefined

		// Get morph settings
		const morphApiKey = this.voidSettingsService.state.globalSettings.morphApiKey
		const hasMorphApiKey = !!morphApiKey

		let enableMorphFastContext = this.voidSettingsService.state.globalSettings.enableMorphFastContext
		if (modelSelection) {
			const modelSelectionOptions = this.voidSettingsService.state.optionsOfModelSelection['Chat'][modelSelection.providerName]?.[modelSelection.modelName]
			if (modelSelectionOptions?.morphFastContext !== undefined) {
				enableMorphFastContext = modelSelectionOptions.morphFastContext
			}
		}

		// Ensure key is present
		enableMorphFastContext = enableMorphFastContext && hasMorphApiKey

		// Get media generation setting
		const enableMediaGeneration = this.voidSettingsService.state.globalSettings.enableMediaGeneration

		const systemMessage = chat_systemMessage({ workspaceFolders, openedURIs, directoryStr, activeURI, persistentTerminalIDs, chatMode, mcpTools, composioTools, acpTools, specialToolFormat: specialToolFormat as any, studentLevel, enableMorphFastContext, enableMediaGeneration, allowedTools, allowExternalTools })
		return systemMessage
	}

	/**
	 * Get Composio tools for the agent.
	 * Uses Tool Router meta tools for self-managed connections and tool discovery.
	 */
	private async _getComposioTools(): Promise<InternalToolInfo[] | undefined> {
		const apiKey = this.voidSettingsService.state.globalSettings.composioApiKey
		if (!apiKey) {
			return undefined
		}

		try {
			if (!this.composioService.isConfigured()) {
				return undefined
			}

			// Create or get a Tool Router session
			const sessionId = this.composioService.getSessionId()
			let session

			if (sessionId) {
				// Use existing session
				session = { session_id: sessionId }
			} else {
				// Create new session
				const enabledToolkits = this.voidSettingsService.state.globalSettings.composioEnabledToolkits
				session = await this.composioService.createSession(
					enabledToolkits.length > 0 ? enabledToolkits : undefined
				)
			}

			// Get meta tools from the session
			const metaTools = await this.composioService.getMetaTools(session.session_id)

			// Convert meta tools to InternalToolInfo format
			return metaTools.map(tool => {
				const params: { [paramName: string]: { description: string } } = {}
				for (const [paramName, param] of Object.entries(tool.parameters.properties)) {
					params[paramName] = {
						description: param.description || `Parameter ${paramName}`,
					}
				}

				return {
					name: tool.name,
					description: tool.description,
					params,
					mcpServerName: 'composio_tool_router',
					// Carry the full Composio parameters schema (with nested properties
					// and `required`) so GBNF-based local servers can sanitize it into a
					// grammar-safe shape instead of losing all type info.
					rawInputSchema: tool.parameters as Record<string, unknown>,
				}
			})
		} catch (err) {
			console.error('[ConvertToLLMMessageService] Error getting Composio tools:', err)
			return undefined
		}
	}




	// --- LLM Chat messages ---

	private _chatMessagesToSimpleMessages(chatMessages: ChatMessage[]): SimpleLLMMessage[] {
		const simpleLLMMessages: SimpleLLMMessage[] = []

		for (const m of chatMessages) {
			if (m.role === 'checkpoint') continue
			if (m.role === 'interrupted_streaming_tool') continue
			if (m.role === 'assistant') {
				simpleLLMMessages.push({
					role: m.role,
					content: m.displayContent,
					reasoning: m.reasoning,
					anthropicReasoning: m.anthropicReasoning,
				})
			}
			else if (m.role === 'tool') {
				simpleLLMMessages.push({
					role: m.role,
					content: m.content,
					name: m.name,
					id: m.id,
					rawParams: m.rawParams,
					thought_signature: m.thought_signature,
				})
			}
			else if (m.role === 'user') {
				simpleLLMMessages.push({
					role: m.role,
					content: m.content,
				})
			}
		}
		return simpleLLMMessages
	}

	prepareLLMSimpleMessages: IConvertToLLMMessageService['prepareLLMSimpleMessages'] = ({ simpleMessages, systemMessage, modelSelection, featureName }) => {
		if (modelSelection === null) return { messages: [], separateSystemMessage: undefined }

		const { overridesOfModel } = this.voidSettingsService.state

		const { providerName, modelName } = modelSelection
		const {
			specialToolFormat,
			contextWindow,
			supportsSystemMessage,
		} = getModelCapabilities(providerName, modelName, overridesOfModel)

		// For models without native tool calling (XML tool calling), use OpenAI-style as default for message formatting
		const ensuredSpecialToolFormat: 'openai-style' | 'anthropic-style' | 'gemini-style' | 'marker-style' = specialToolFormat || 'openai-style'

		const modelSelectionOptions = this.voidSettingsService.state.optionsOfModelSelection[featureName][modelSelection.providerName]?.[modelSelection.modelName]

		// Get combined AI instructions
		const aiInstructions = this._getCombinedAIInstructions();

		const isReasoningEnabled = getIsReasoningEnabledState(featureName, providerName, modelName, modelSelectionOptions, overridesOfModel)
		const reservedOutputTokenSpace = getReservedOutputTokenSpace(providerName, modelName, { isReasoningEnabled, overridesOfModel })

		const { messages, separateSystemMessage } = prepareMessages({
			messages: simpleMessages,
			systemMessage,
			aiInstructions,
			supportsSystemMessage,
			specialToolFormat: ensuredSpecialToolFormat,
			supportsAnthropicReasoning: providerName === 'anthropic' || providerName === 'gemini' || ensuredSpecialToolFormat === 'gemini-style',
			contextWindow,
			reservedOutputTokenSpace,
			providerName,
		})
		return { messages, separateSystemMessage };
	}
	buildSubagentSystemMessage: IConvertToLLMMessageService['buildSubagentSystemMessage'] = async ({ rolePrompt, modelSelection, allowedTools, allowExternalTools }) => {
		// Derive the model's tool-calling style so chat_systemMessage emits the right
		// tool-description block (XML/marker text vs native-tool rules). The caller
		// guards against a null modelSelection before running a subagent; the
		// fallbacks here are purely defensive.
		const providerName = modelSelection?.providerName ?? 'anthropic'
		const modelName = modelSelection?.modelName ?? ''
		const { specialToolFormat } = getModelCapabilities(providerName, modelName, this.voidSettingsService.state.overridesOfModel)
		const base = await this._generateChatMessagesSystemMessage('code', specialToolFormat, modelSelection, allowedTools, allowExternalTools)
		return `${rolePrompt}\n\n${base}`
	}
	prepareLLMChatMessages: IConvertToLLMMessageService['prepareLLMChatMessages'] = async ({ chatMessages, chatMode, modelSelection, loadedSkills, orchestrationResult, compaction, usageAnchor }) => {
		if (modelSelection === null) return { messages: [], separateSystemMessage: undefined, tokenUsage: { used: 0, total: 0, percentage: 0 }, compressionStats: undefined }

		const { overridesOfModel } = this.voidSettingsService.state

		const { providerName, modelName } = modelSelection
		const {
			specialToolFormat,
			contextWindow,
			supportsSystemMessage,
		} = getModelCapabilities(providerName, modelName, overridesOfModel)

		// For models without native tool calling (XML tool calling), use OpenAI-style as default for message formatting
		// The actual tool calling will be handled via XML by the provider
		const ensuredSpecialToolFormat: 'openai-style' | 'anthropic-style' | 'gemini-style' | 'marker-style' = specialToolFormat || 'openai-style'

		const { disableSystemMessage } = this.voidSettingsService.state.globalSettings;
		// Pass the ACTUAL specialToolFormat (can be undefined) to system message for XML tool calling
		const fullSystemMessage = await this._generateChatMessagesSystemMessage(chatMode, specialToolFormat, modelSelection)
		
		// Get combined AI instructions
		const aiInstructions = this._getCombinedAIInstructions();

		// Add loaded skills to system message
		let systemMessage = disableSystemMessage ? '' : fullSystemMessage;
		if (loadedSkills && Object.keys(loadedSkills).length > 0) {
			const skillsText = Object.entries(loadedSkills)
				.map(([name, instructions]) => `### SKILL: ${name}\n${instructions}`)
				.join('\n\n');
			systemMessage += `\n\n## LOADED SKILLS\nYou have loaded the following specialized skills for this conversation. Adhere to their instructions and patterns:\n\n${skillsText}`;
		}

		// Add orchestration suggestions to system message
		if (orchestrationResult && orchestrationResult.suggestions.length > 0) {
			const suggestionsText = orchestrationResult.suggestions
				.map(s => `- **${s.toolName}**: ${s.toolParams ? JSON.stringify(s.toolParams) : 'no params specified'} (confidence: ${s.confidence}) - ${s.reasoning}`)
				.join('\n');
			systemMessage += `\n\n## TOOL ORCHESTRATION SUGGESTIONS\nThe tool orchestration model has suggested the following tools for this request. These are **suggestions only** - you should use them as guidance but use your own judgment to determine the best approach.\n\nSummary: ${orchestrationResult.summary}\n\nSuggested tools:\n${suggestionsText}\n\nOrchestration reasoning: ${orchestrationResult.reasoning}`;
		}

		const modelSelectionOptions = this.voidSettingsService.state.optionsOfModelSelection['Chat'][modelSelection.providerName]?.[modelSelection.modelName]

		const isReasoningEnabled = getIsReasoningEnabledState('Chat', providerName, modelName, modelSelectionOptions, overridesOfModel)
		const reservedOutputTokenSpace = getReservedOutputTokenSpace(providerName, modelName, { isReasoningEnabled, overridesOfModel })
		// Honor a /compact snapshot: replace the leading `compactedChatMessageCount`
		// messages with a single user-role summary message, then convert only the
		// trailing (kept) messages. The original ChatMessages are retained by the
		// caller for rewind — only this LLM-facing view is compressed. A user-role
		// summary is the only mid-stream role every provider accepts; the summary
		// text is framed as background context, not a fresh user request.
		let llmMessages: SimpleLLMMessage[]
		if (compaction && compaction.compactedChatMessageCount > 0 && chatMessages.length > compaction.compactedChatMessageCount) {
			const kept = chatMessages.slice(compaction.compactedChatMessageCount)
			llmMessages = this._chatMessagesToSimpleMessages(kept)
			llmMessages.unshift({ role: 'user', content: compaction.summaryText })
		} else {
			llmMessages = this._chatMessagesToSimpleMessages(chatMessages)
		}

		let { messages, separateSystemMessage } = prepareMessages({
			messages: llmMessages,
			systemMessage,
			aiInstructions,
			supportsSystemMessage,
			specialToolFormat: ensuredSpecialToolFormat,
			supportsAnthropicReasoning: providerName === 'anthropic' || providerName === 'gemini' || ensuredSpecialToolFormat === 'gemini-style',
			contextWindow,
			reservedOutputTokenSpace,
			providerName,
		})

		// Apply context window compression using rolling window approach
		const fullModelName = `${providerName}:${modelName}`;
		const contextWindowSize = contextWindow;
		
		// Get reserved tokens calculation
		const effectiveContext = contextWindowSize - (reservedOutputTokenSpace ?? 4096);

		// Count tokens before compression. When a real usage anchor from the last
		// successful LLM call is available and still valid, anchor on its measured
		// promptTokens and only cheaply estimate the trailing messages — avoiding a
		// full re-tokenization of the entire thread (IPC count) on every call.
		const anchor = usageAnchor;
		const anchorUsable = !!anchor
			&& anchor.providerName === providerName
			&& anchor.modelName === modelName
			&& anchor.promptTokens > 0
			&& (compaction?.compactedChatMessageCount ?? 0) === anchor.compactedChatMessageCountAtAnchor
			&& anchor.coveredThreadMessageCount <= chatMessages.length;

		let tokenCount: number;
		if (anchorUsable && anchor) {
			const trailingChatMessages = chatMessages.slice(anchor.coveredThreadMessageCount);
			const trailingSimple = this._chatMessagesToSimpleMessages(trailingChatMessages);
			const trailingTokens = this.tokenCountingService.estimateMessagesTokensFast(trailingSimple);
			tokenCount = anchor.promptTokens + trailingTokens;
			console.log(`[ConvertToLLMMessageService] Token usage (usage-anchored): ${anchor.promptTokens} + ${trailingTokens} trailing = ${tokenCount}/${effectiveContext} for ${providerName}/${modelName}`);
		} else {
			try {
				tokenCount = await this.tokenCountingService.countMessagesTokensAsync(messages, fullModelName);
			} catch (error) {
				// Fallback to estimate
				tokenCount = Math.ceil(JSON.stringify(messages).length / 4);
			}
		}

		const usage = tokenCount / effectiveContext;
		if (!anchorUsable) {
			console.log(`[ConvertToLLMMessageService] Token usage: ${tokenCount}/${effectiveContext} (${(usage * 100).toFixed(1)}%) for ${providerName}/${modelName}`);
		}

		// Compress if using more than 70% of effective context (more aggressive threshold)
		// Lower threshold = compress earlier = more reliable prevention of overflow.
		// Uses the tokenCount computed above (usage-anchored or fully counted) rather
		// than triggering a second full re-count inside compressionService.
		const compressionThreshold = 0.70; // 70% instead of 80%
		const needsCompression = usage > compressionThreshold;

		let compressionStats = undefined;
		if (needsCompression) {
			console.log(`[ConvertToLLMMessageService] Context window usage high (${(usage * 100).toFixed(1)}%), applying rolling window compression...`);

			const { compressedMessages, stats } = await this.compressionService.compressMessages(
				messages,
				fullModelName,
				{
					targetUsage: 0.80, // Target 80% of effective context
					keepLastNMessages: 10, // Keep more recent messages
					enableSummarization: true,
					maxToolResultLength: 1500, // Truncate tool results more aggressively
					reservedTokens: reservedOutputTokenSpace ?? 4096,
					emergencyKeepLastN: 4,
				}
			);

			compressionStats = {
				originalMessageCount: stats.originalMessageCount,
				finalMessageCount: stats.finalMessageCount,
				originalTokens: stats.originalTokens,
				finalTokens: stats.finalTokens,
				compressionRatio: stats.compressionRatio,
				messagesRemoved: stats.messagesRemoved,
				messagesSummarized: stats.messagesSummarized,
			};

			console.log(`[ConvertToLLMMessageService] Compression complete: ${stats.originalMessageCount} → ${stats.finalMessageCount} messages, ${stats.originalTokens} → ${stats.finalTokens} tokens (${stats.compressionRatio}% of original), removed ${stats.messagesRemoved}, summarized ${stats.messagesSummarized}`);

			// Update token count after compression
			try {
				tokenCount = await this.tokenCountingService.countMessagesTokensAsync(compressedMessages, fullModelName);
			} catch {
				tokenCount = stats.finalTokens;
			}

			messages = compressedMessages;
		}

		// Final safety check - if still significantly over, log warning
		const finalUsage = tokenCount / effectiveContext;
		if (finalUsage > 0.95) {
			console.warn(`[ConvertToLLMMessageService] WARNING: Still at ${(finalUsage * 100).toFixed(1)}% of context after compression. Consider reducing conversation length.`);
		}

		// Return messages with token usage info
		return {
			messages,
			separateSystemMessage,
			tokenUsage: {
				used: tokenCount,
				total: contextWindowSize,
				percentage: usage * 100
			},
			compressionStats
		};
	}


	// --- FIM ---

	prepareFIMMessage: IConvertToLLMMessageService['prepareFIMMessage'] = ({ messages }) => {
		// Get combined AI instructions with the provided aiInstructions as the base
		const combinedInstructions = this._getCombinedAIInstructions();

		let prefix = `\
${!combinedInstructions ? '' : `\
// Instructions:
// Do not output an explanation. Try to avoid outputting comments. Only output the middle code.
${combinedInstructions.split('\n').map(line => `//${line}`).join('\n')}`}

${messages.prefix}`

		const suffix = messages.suffix
		const stopTokens = messages.stopTokens
		return { prefix, suffix, stopTokens }
	}

	updateTokenRatio(modelName: string, estimatedTokens: number, actualTokens: number): void {
		this.tokenCountingService.updateTokenRatio(modelName, estimatedTokens, actualTokens);
	}


}


registerSingleton(IConvertToLLMMessageService, ConvertToLLMMessageService, InstantiationType.Delayed);








/*
Gemini has this, but they're openai-compat so we don't need to implement this
gemini request:
{   "role": "assistant",
	"content": null,
	"function_call": {
		"name": "get_weather",
		"arguments": {
			"latitude": 48.8566,
			"longitude": 2.3522
		}
	}
}

gemini response:
{   "role": "assistant",
	"function_response": {
		"name": "get_weather",
			"response": {
			"temperature": "15°C",
				"condition": "Cloudy"
		}
	}
}
*/



