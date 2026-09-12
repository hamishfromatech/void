/*--------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

// disable foreign import complaints
/* eslint-disable */
import Anthropic from '@anthropic-ai/sdk';
import { Ollama } from 'ollama';
import OpenAI, { ClientOptions, AzureOpenAI } from 'openai';
import { MistralCore } from '@mistralai/mistralai/core.js';
import { fimComplete } from '@mistralai/mistralai/funcs/fimComplete.js';
import { Tool as GeminiTool, FunctionDeclaration, GoogleGenAI, ThinkingConfig, Schema, Type } from '@google/genai';
import { GoogleAuth } from 'google-auth-library'
/* eslint-enable */

import { AnthropicLLMChatMessage, AnthropicReasoning, GeminiLLMChatMessage, LLMChatMessage, LLMFIMMessage, ModelListParams, OllamaModelResponse, OnError, OnFinalMessage, OnText, OpenAILLMChatMessage, RawToolCallObj, RawToolParamsObj } from '../../common/sendLLMMessageTypes.js';
import { ChatMode, displayInfoOfProviderName, GlobalSettings, ModelSelectionOptions, OverridesOfModel, ProviderName, SettingsOfProvider } from '../../common/voidSettingsTypes.js';
import { getSendableReasoningInfo, getModelCapabilities, getProviderCapabilities, defaultProviderSettings, getReservedOutputTokenSpace } from '../../common/modelCapabilities.js';
import { availableTools, InternalToolInfo } from '../../common/prompt/prompts.js';
import { sanitizeJsonSchemaForGBNF } from '../../common/helpers/sanitizeJsonSchemaForGBNF.js';
import { generateUuid } from '../../../../../base/common/uuid.js';
import { voidDevLog, voidDevWarn } from '../../common/devLog.js';
import product from '../../../../../platform/product/common/product.js';

/**
 * Read a provider-specific reasoning field from an OpenAI-compatible streaming
 * delta. The SDK's `ChatCompletionChunk.Choice.Delta` type only declares
 * `content`/`role`/`tool_calls`/`function`/`refusal`, but providers (Ollama,
 * DeepSeek, etc.) add `reasoning`/`thinking`/custom fields. We narrow the delta
 * to a string-keyed record instead of `as any` so the access stays type-safe.
 * Mirrors the original `delta?.[field] || delta?.reasoning || delta?.thinking`
 * truthy-fallback semantics, then coerces to string (the old `+ ''`).
 *
 * The fallback chain covers the three field names OpenAI-compatible servers use
 * for separated reasoning: `reasoning` (gpt-oss / xAI), `reasoning_content`
 * (DeepSeek, LM Studio, LiteLLM, OpenAI-compatible), and `thinking` (Ollama).
 * The configured `fieldName` is tried first so a provider's declared field wins
 * when present; the rest catch models whose field differs from the provider default
 * (e.g. LM Studio streams `reasoning_content` for Qwen even though lmStudio is
 * configured with `nameOfFieldInDelta: 'reasoning'` for gpt-oss).
 */
const readReasoningFromDelta = (delta: object, fieldName: string | undefined): string => {
	const d = delta as Record<string, unknown>
	const v = (fieldName ? d[fieldName] : undefined) || d['reasoning'] || d['reasoning_content'] || d['thinking'] || ''
	return typeof v === 'string' ? v : (v === null || v === undefined) ? '' : String(v)
}

/**
 * Gemini streaming chunks expose `text`/`functionCalls` in the `@google/genai`
 * typings, but `candidates` (finish-reason + thought parts) and the non-standard
 * `thought_signature` on function calls aren't typed. Narrow to these shapes
 * instead of `as any`.
 */
interface GeminiStreamChunkExtras {
	candidates?: Array<{
		finishReason?: string
		content?: { parts?: Array<{ thought?: boolean, text?: string }> }
	}>
	usageMetadata?: {
		promptTokenCount?: number
		candidatesTokenCount?: number
		totalTokenCount?: number
	}
}
interface GeminiFunctionCallExtras {
	thought_signature?: string
}

const getGoogleApiKey = async () => {
	// module‑level singleton
	const auth = new GoogleAuth({ scopes: `https://www.googleapis.com/auth/cloud-platform` });
	const key = await auth.getAccessToken()
	if (!key) throw new Error(`Google API failed to generate a key.`)
	return key
}

// DISABLED: A-Coder OAuth provider - commented out to prevent memory leaks
// // A-Coder session token - obtained from OAuth authentication
// // The session token is used to authenticate with A-Coder's backend
// // which proxies requests to chutes.ai with the master API key (user never sees it)
// let _aCoderSessionToken: string | null = null
//
// /**
//  * A-Coder backend URL - proxies requests to chutes.ai with the master API key
//  * Can be overridden via environment variable for development
//  */
// const ACODER_API_URL = process.env.ACODER_API_URL || 'https://api.a-coder.dev/v1'
//
// /**
//  * Get the A-Coder session token.
//  * The token is obtained via OAuth authentication and stored securely.
//  * For development, it can be set via ACODER_SESSION_TOKEN environment variable.
//  */
// export const getACoderSessionToken = (): string => {
// 	// First check if we have a cached token
// 	if (_aCoderSessionToken) {
// 		return _aCoderSessionToken
// 	}
// 	// Fall back to environment variable for development
// 	const envToken = process.env.ACODER_SESSION_TOKEN || ''
// 	if (envToken) {
// 		_aCoderSessionToken = envToken
// 	}
// 	return _aCoderSessionToken || ''
// }
//
// /**
//  * Set the A-Coder session token (called by OAuth service after successful authentication)
//  */
// export const setACoderSessionToken = (token: string) => {
// 	_aCoderSessionToken = token
// }
//
// /**
//  * Clear the A-Coder session token (called on sign out)
//  */
// export const clearACoderSessionToken = () => {
// 	_aCoderSessionToken = null
// }
//
// // Legacy exports for backwards compatibility
// export const getACoderApiKey = getACoderSessionToken
// export const setACoderApiKey = setACoderSessionToken
// export const clearACoderApiKey = clearACoderSessionToken


type InternalCommonMessageParams = {
	onText: OnText;
	onFinalMessage: OnFinalMessage;
	onError: OnError;
	providerName: ProviderName;
	settingsOfProvider: SettingsOfProvider;
	globalSettings: GlobalSettings;
	modelSelectionOptions: ModelSelectionOptions | undefined;
	overridesOfModel: OverridesOfModel | undefined;
	modelName: string;
	_setAborter: (aborter: () => void) => void;
}

type SendChatParams_Internal = InternalCommonMessageParams & {
	messages: LLMChatMessage[];
	separateSystemMessage: string | undefined;
	chatMode: ChatMode | null;
	mcpTools: InternalToolInfo[] | undefined;
	acpTools?: InternalToolInfo[] | undefined;
	composioTools: InternalToolInfo[] | undefined;
	allowedTools?: string[];
	allowExternalTools?: boolean;
}
type SendFIMParams_Internal = InternalCommonMessageParams & { messages: LLMFIMMessage; separateSystemMessage: string | undefined; }
export type ListParams_Internal<ModelResponse> = ModelListParams<ModelResponse>


// Helper to parse partial JSON for streaming UI
const parsePartialJSON = (jsonStr: string): Record<string, any> => {
	if (!jsonStr) return {}
	try {
		// Try full parse first
		return JSON.parse(jsonStr)
	} catch (e) {
		// If it fails, it might be partial. 
		// We try to close it if it's an object
		const trimmed = jsonStr.trim()
		if (trimmed.startsWith('{')) {
			try {
				// Very simple attempt to close braces
				return JSON.parse(trimmed + '}')
			} catch (e2) {
				try {
					return JSON.parse(trimmed + '"}')
				} catch (e3) {
					return {}
				}
			}
		}
		return {}
	}
}


const invalidApiKeyMessage = (providerName: ProviderName) => `Invalid ${displayInfoOfProviderName(providerName).title} API key.`

// ------------ OPENAI-COMPATIBLE (HELPERS) ------------



const parseHeadersJSON = (s: string | undefined): Record<string, string | null | undefined> | undefined => {
	if (!s) return undefined
	try {
		return JSON.parse(s)
	} catch (e) {
		throw new Error(`Error parsing OpenAI-Compatible headers: ${s} is not a valid JSON.`)
	}
}

const newOpenAICompatibleSDK = async ({ settingsOfProvider, providerName, includeInPayload }: { settingsOfProvider: SettingsOfProvider, providerName: ProviderName, includeInPayload?: { [s: string]: any } }) => {
	const commonPayloadOpts: ClientOptions = {
		dangerouslyAllowBrowser: true,
		...includeInPayload,
	}
	if (providerName === 'openAI') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ apiKey: thisConfig.apiKey, ...commonPayloadOpts })
	}
	else if (providerName === 'ollama') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({
			baseURL: `${thisConfig.endpoint}/v1`,
			apiKey: 'noop',
			// Add specific configurations for better Ollama compatibility
			defaultHeaders: {
				'HTTP-User-Agent': `A-Coder-IDE/${product.voidVersion || product.version}`
			},
			// Increase timeout for Ollama models which can be slower
			timeout: 120000, // 2 minutes
			...commonPayloadOpts
		})
	}
	else if (providerName === 'ollamaCloud') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({
			baseURL: `${thisConfig.endpoint}/v1`,
			apiKey: thisConfig.apiKey,
			// Add specific configurations for better Ollama compatibility
			defaultHeaders: {
				'HTTP-User-Agent': `A-Coder-IDE/${product.voidVersion || product.version}`
			},
			// Increase timeout for Ollama models which can be slower
			timeout: 120000, // 2 minutes
			...commonPayloadOpts
		})
	}
	else if (providerName === 'vLLM') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ baseURL: `${thisConfig.endpoint}/v1`, apiKey: 'noop', ...commonPayloadOpts })
	}
	else if (providerName === 'liteLLM') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ baseURL: `${thisConfig.endpoint}/v1`, apiKey: 'noop', ...commonPayloadOpts })
	}
	else if (providerName === 'lmStudio') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ baseURL: `${thisConfig.endpoint}/v1`, apiKey: 'noop', ...commonPayloadOpts })
	}
	else if (providerName === 'openRouter') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({
			baseURL: 'https://openrouter.ai/api/v1',
			apiKey: thisConfig.apiKey,
			defaultHeaders: {
				'HTTP-Referer': 'https://a-coder.dev', // Optional, for including your app on openrouter.ai rankings.
				'X-Title': 'A-Coder IDE', // Optional. Shows in rankings on openrouter.ai.
			},
			...commonPayloadOpts,
		})
	}
	else if (providerName === 'googleVertex') {
		// https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library
		const thisConfig = settingsOfProvider[providerName]
		const baseURL = `https://${thisConfig.region}-aiplatform.googleapis.com/v1/projects/${thisConfig.project}/locations/${thisConfig.region}/endpoints/${'openapi'}`
		const apiKey = await getGoogleApiKey()
		return new OpenAI({ baseURL: baseURL, apiKey: apiKey, ...commonPayloadOpts })
	}
	else if (providerName === 'microsoftAzure') {
		// https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-chat-completions/get-chat-completions?view=rest-aifoundry-model-inference-2024-05-01-preview&tabs=HTTP
		//  https://github.com/openai/openai-node?tab=readme-ov-file#microsoft-azure-openai
		const thisConfig = settingsOfProvider[providerName]
		const endpoint = `https://${thisConfig.project}.openai.azure.com/`;
		const apiVersion = thisConfig.azureApiVersion ?? '2024-04-01-preview';
		const options = { endpoint, apiKey: thisConfig.apiKey, apiVersion };
		return new AzureOpenAI({ ...options, ...commonPayloadOpts });
	}
	else if (providerName === 'awsBedrock') {
		/**
		  * We treat Bedrock as *OpenAI-compatible only through a proxy*:
		  *   • LiteLLM default → http://localhost:4000/v1
		  *   • Bedrock-Access-Gateway → https://<api-id>.execute-api.<region>.amazonaws.com/openai/
		  *
		  * The native Bedrock runtime endpoint
		  *   https://bedrock-runtime.<region>.amazonaws.com
		  * is **NOT** OpenAI-compatible, so we do *not* fall back to it here.
		  */
		const { endpoint, apiKey } = settingsOfProvider.awsBedrock

		// ① use the user-supplied proxy if present
		// ② otherwise default to local LiteLLM
		let baseURL = endpoint || 'http://localhost:4000/v1'

		// Normalize: make sure we end with “/v1”
		if (!baseURL.endsWith('/v1'))
			baseURL = baseURL.replace(/\/+$/, '') + '/v1'

		return new OpenAI({ baseURL, apiKey, ...commonPayloadOpts })
	}


	else if (providerName === 'deepseek') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ baseURL: 'https://api.deepseek.com/v1', apiKey: thisConfig.apiKey, ...commonPayloadOpts })
	}
	else if (providerName === 'openAICompatible') {
		const thisConfig = settingsOfProvider[providerName]
		const headers = parseHeadersJSON(thisConfig.headersJSON)
		return new OpenAI({ baseURL: thisConfig.endpoint, apiKey: thisConfig.apiKey, defaultHeaders: headers, ...commonPayloadOpts })
	}
	else if (providerName === 'groq') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ baseURL: 'https://api.groq.com/openai/v1', apiKey: thisConfig.apiKey, ...commonPayloadOpts })
	}
	else if (providerName === 'xAI') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ baseURL: 'https://api.x.ai/v1', apiKey: thisConfig.apiKey, ...commonPayloadOpts })
	}
	else if (providerName === 'mistral') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({ baseURL: 'https://api.mistral.ai/v1', apiKey: thisConfig.apiKey, ...commonPayloadOpts })
	}
	else if (providerName === 'aCoder') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({
			// LLM inference proxy. See voidSettingsTypes.ts (aCoder key link) for the
			// source-of-truth note on A-Coder's three hostnames.
			baseURL: 'https://provider.atech.industries/v1',
			apiKey: thisConfig.apiKey,
			defaultHeaders: {
				'HTTP-User-Agent': `A-Coder-IDE/${product.voidVersion || product.version}`,
			},
			...commonPayloadOpts
		})
	}
	else if (providerName === 'openAdapter') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({
			baseURL: 'https://api.openadapter.in/v1',
			apiKey: thisConfig.apiKey,
			...commonPayloadOpts
		})
	}
	else if (providerName === 'llamaCpp') {
		const thisConfig = settingsOfProvider[providerName]
		return new OpenAI({
			baseURL: thisConfig.endpoint + '/v1',
			apiKey: 'no-key-needed', // llama.cpp doesn't require an API key
			...commonPayloadOpts
		})
	}

	else throw new Error(`Provider "${providerName}" is not supported.`)
}


// llama.cpp exposes a dedicated /infill endpoint for fill-in-the-middle that applies the
// model's FIM special tokens (FIM_PRE/FIM_SUF/FIM_MID). The OpenAI-compatible /v1/completions
// path with a `suffix` doesn't tokenize FIM correctly, so route llama.cpp FIM here.
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#postinfill
// Non-streaming (matches the OpenAI-compat FIM behavior): the response `content` holds the
// full infill. `/infill` accepts the same options as `/completion` (n_predict, stop, ...).
const _sendLlamaCppInfillFIM = async ({ prefix, suffix, stopTokens, onFinalMessage, onError, settingsOfProvider, _setAborter }: { prefix: string, suffix: string, stopTokens?: string[], onFinalMessage: OnFinalMessage, onError: OnError, settingsOfProvider: SettingsOfProvider, _setAborter: (aborter: () => void) => void }) => {
	const endpoint = (settingsOfProvider.llamaCpp.endpoint || '').replace(/\/+$/, '')
	if (!endpoint) {
		onError({ message: `llama.cpp endpoint was empty (set it in A-Coder IDE settings).`, fullError: null })
		return
	}
	const controller = new AbortController()
	_setAborter(() => controller.abort())
	try {
		const res = await fetch(`${endpoint}/infill`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			signal: controller.signal,
			body: JSON.stringify({
				input_prefix: prefix,
				input_suffix: suffix,
				stream: false,
				n_predict: 300,
				...(stopTokens && stopTokens.length > 0 ? { stop: stopTokens } : {}),
			}),
		})
		if (!res.ok) {
			const text = await res.text().catch(() => '')
			onError({ message: `llama.cpp /infill failed (${res.status} ${res.statusText})${text ? `: ${text}` : ''}`, fullError: null })
			return
		}
		const data: { content?: string } = await res.json()
		onFinalMessage({ fullText: typeof data.content === 'string' ? data.content : '', fullReasoning: '', anthropicReasoning: null })
	} catch (error) {
		if (error instanceof Error && error.name === 'AbortError') return
		onError({ message: error + '', fullError: error as Error })
	}
}


const _sendOpenAICompatibleFIM = async ({ messages: { prefix, suffix, stopTokens }, onFinalMessage, onError, settingsOfProvider, globalSettings, modelName: modelName_, _setAborter, providerName, overridesOfModel }: SendFIMParams_Internal) => {

	// llama.cpp: use the native /infill endpoint instead of the OpenAI /v1/completions path.
	// Routed before the supportsFIM gate because autodetected llama.cpp models have unknown FIM
	// support client-side; /infill will apply FIM tokens if the loaded model defines them.
	if (providerName === 'llamaCpp') {
		return _sendLlamaCppInfillFIM({ prefix, suffix, stopTokens, onFinalMessage, onError, settingsOfProvider, _setAborter })
	}

	const {
		modelName,
		supportsFIM,
		additionalOpenAIPayload,
	} = getModelCapabilities(providerName, modelName_, overridesOfModel)

	if (!supportsFIM) {
		if (modelName === modelName_)
			onError({ message: `Model ${modelName} does not support FIM.`, fullError: null })
		else
			onError({ message: `Model ${modelName_} (${modelName}) does not support FIM.`, fullError: null })
		return
	}

	const openai = await newOpenAICompatibleSDK({ providerName, settingsOfProvider, includeInPayload: additionalOpenAIPayload })
	try {
		const response = await openai.completions.create({
			model: modelName,
			prompt: prefix,
			suffix: suffix,
			stop: stopTokens,
			max_tokens: 300,
		})
		const fullText = response.choices[0]?.text
		onFinalMessage({ fullText, fullReasoning: '', anthropicReasoning: null })
	} catch (error) {
		if (error instanceof OpenAI.APIError && error.status === 401) { onError({ message: invalidApiKeyMessage(providerName), fullError: error }); }
		else { onError({ message: error + '', fullError: error }); }
	}
}


// Local llama.cpp-derived servers (llama.cpp, LM Studio, Ollama, Ollama Cloud) constrain
// tool-call output by compiling the `tools` schemas into a GBNF grammar server-side. Their
// `json_schema_to_grammar` is strict and rejects common MCP/Composio schema constructs —
// nullable `type: ["string","null"]` unions, `$ref`, `oneOf`/`anyOf`, nested `items.properties`,
// `format`/`pattern`, empty `enum`, `not`, `if/then/else`, etc. — with
// `Failed to initialize samplers: failed to parse grammar` (HTTP 400), which hard-breaks every
// request that carries tools. (Ollama masks this via its no-tools fallback in
// _sendOllamaChatWithFallback; llama.cpp and LM Studio do not, so they hard-fail.)
//
// For these providers we sanitize the tool's raw JSON-Schema into a GBNF-safe shape
// (sanitizeJsonSchemaForGBNF) instead of flattening every param to `{type:'string'}`. This
// preserves enums, arrays, nested objects, numeric constraints, and `required` where the
// schema declares them, while always compiling. Tools without a raw schema (builtin A-Coder
// tools, ACP agents) still fall back to the all-string shape, since their params carry only a
// description. Cloud / vLLM servers don't use GBNF and benefit from the real schema, so the rich
// `type`/`enum`/`items` passthrough is kept for everyone else.
const GBNF_GRAMMAR_TOOL_PROVIDERS: ReadonlySet<string> = new Set(['llamaCpp', 'lmStudio', 'ollama', 'ollamaCloud'])

const toOpenAICompatibleTool = (toolInfo: InternalToolInfo, providerName?: string) => {
	const { name, description, params } = toolInfo
	const isGbnfProvider = !!providerName && GBNF_GRAMMAR_TOOL_PROVIDERS.has(providerName)

	// GBNF-based local servers: prefer the raw input schema (full JSON-Schema with
	// `required`, nested `properties`, `$ref`, `anyOf`) and sanitize it into a
	// grammar-safe shape. Builtin tools / ACP agents have no raw schema, so they fall
	// back to the all-string params (current behavior). The sanitized schema's
	// per-property `description` comes from the raw schema verbatim (the flattened
	// `params[key].description` is JSON-stringified and less useful, so we don't merge it).
	if (isGbnfProvider && toolInfo.rawInputSchema) {
		const sanitized = sanitizeJsonSchemaForGBNF(toolInfo.rawInputSchema)
		return {
			type: 'function',
			function: {
				name: name,
				// strict: true, // strict mode - https://platform.openai.com/docs/guides/function-calling?api-mode=chat
				description: description,
				parameters: sanitized,
			}
		} satisfies OpenAI.Chat.Completions.ChatCompletionTool
	}

	const paramsWithType: { [s: string]: { description: string; type: string; enum?: unknown[]; items?: unknown } } = {}
	for (const key in params) {
		if (isGbnfProvider) {
			// No raw schema available (builtin / ACP): flatten to the safe string-only shape
			// so the grammar always compiles. `description` is preserved; rich types are dropped.
			paramsWithType[key] = { description: params[key].description, type: 'string' }
		} else {
			// Emit the real JSON-Schema type/enum/items when the tool carries them (MCP/Composio
			// tools whose inputSchema declares them), falling back to `type: 'string'` for
			// builtin A-Coder tools (whose params are all strings). `enum`/`items` pass through via the spread.
			paramsWithType[key] = { ...params[key], type: params[key].type ?? 'string' }
		}
	}

	return {
		type: 'function',
		function: {
			name: name,
			// strict: true, // strict mode - https://platform.openai.com/docs/guides/function-calling?api-mode=chat
			description: description,
			parameters: {
				type: 'object',
				properties: paramsWithType, // \u{2705} FIX: Use paramsWithType instead of params to include type field for llama.cpp compatibility
				// required: Object.keys(params), // in strict mode, all params are required and additionalProperties is false
				// additionalProperties: false,
			},
		}
	} satisfies OpenAI.Chat.Completions.ChatCompletionTool
}

const openAITools = (chatMode: ChatMode | null, mcpTools: InternalToolInfo[] | undefined, acpTools: InternalToolInfo[] | undefined, composioTools: InternalToolInfo[] | undefined, options?: { enableMorphFastContext?: boolean; enableMediaGeneration?: boolean; allowedTools?: string[]; allowExternalTools?: boolean; providerName?: string }) => {
	const allowedTools = availableTools(chatMode, mcpTools, composioTools, acpTools, options)
	if (!allowedTools || Object.keys(allowedTools).length === 0) return null

	const openAITools: OpenAI.Chat.Completions.ChatCompletionTool[] = []
	for (const t in allowedTools ?? {}) {
		openAITools.push(toOpenAICompatibleTool(allowedTools[t], options?.providerName))
	}
	return openAITools
}


// Find the end of a JSON object, handling nested braces
const findJsonObjectEnd = (str: string): number => {
	let depth = 0
	let inString = false
	let escape = false

	for (let i = 0; i < str.length; i++) {
		const char = str[i]

		if (escape) {
			escape = false
			continue
		}

		if (char === '\\' && inString) {
			escape = true
			continue
		}

		if (char === '"') {
			inString = !inString
			continue
		}

		if (inString) continue

		if (char === '{') depth++
		if (char === '}') {
			depth--
			if (depth === 0) return i
		}
	}
	return -1
}

// convert LLM tool call to our tool format
const rawToolCallObjOfParamsStr = (name: string, toolParamsStr: string, id: string, thought_signature?: string): RawToolCallObj | null => {
	if (!toolParamsStr) {
		voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} Tool call "${name}" has empty parameters string`)
		return null
	}

	let input: unknown
	try {
		input = JSON.parse(toolParamsStr)
	}
	catch (e) {
		// Try to handle concatenated JSON objects like {"uri":"a"}{"uri":"b"}
		// This happens when LLMs try to call multiple tools at once incorrectly
		const firstObjectEnd = findJsonObjectEnd(toolParamsStr)
		if (firstObjectEnd !== -1 && firstObjectEnd < toolParamsStr.length - 1) {
			const firstObject = toolParamsStr.substring(0, firstObjectEnd + 1)
			try {
				input = JSON.parse(firstObject)
				voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} LLM sent concatenated tool calls, extracting first one for "${name}"`)
			} catch (e2) {
				// Fallback to partial JSON parser for very malformed but mostly complete JSON
				input = parsePartialJSON(toolParamsStr)
				if (Object.keys(input as object).length === 0) {
					voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} Failed to parse tool parameters for "${name}":`, e)
					return null
				}
			}
		} else {
			// Fallback to partial JSON parser
			input = parsePartialJSON(toolParamsStr)
			if (Object.keys(input as object).length === 0) {
				voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} Failed to parse tool parameters for "${name}":`, e)
				return null
			}
		}
	}

	if (input === null || typeof input !== 'object') {
		voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} Tool call "${name}" parsed to invalid type:`, typeof input)
		return null
	}

	const rawParams: RawToolParamsObj = input as RawToolParamsObj
	voidDevLog(`[sendLLMMessage] ✓ Successfully parsed tool call "${name}" with ${Object.keys(rawParams).length} parameters`)
	return { id, name, rawParams, doneParams: Object.keys(rawParams), isDone: true, thought_signature }
}


const rawToolCallObjOfAnthropicParams = (toolBlock: Anthropic.Messages.ToolUseBlock): RawToolCallObj | null => {
	const { id, name, input } = toolBlock

	if (input === null) return null
	if (typeof input !== 'object') return null

	const rawParams: RawToolParamsObj = input
	return { id, name, rawParams, doneParams: Object.keys(rawParams), isDone: true }
}


// Sanitize messages for OpenAI-compatible providers that are strict about
// schema validation (e.g. Mistral, and OpenRouter when routing to Mistral).
// Strips non-standard fields like `reasoning`, `thought_signature`, and `name`
// on `role: 'tool'` messages that were added for internal tracking or Gemini
// proxy compatibility.
const sanitizeOpenAIMessages = (messages: any[]): any[] => {
	return messages.map(m => {
		if (m.role === 'assistant') {
			const sanitized: any = {
				role: m.role,
				content: m.content,
			}
			if (m.tool_calls) {
				sanitized.tool_calls = m.tool_calls.map((tc: any) => ({
					type: tc.type,
					id: tc.id,
					function: {
						name: tc.function?.name,
						arguments: tc.function?.arguments,
					}
				}))
			}
			return sanitized
		}
		if (m.role === 'tool') {
			return {
				role: m.role,
				content: m.content,
				tool_call_id: m.tool_call_id,
			}
		}
		return m
	})
}

// Ollama Cloud is stricter than local Ollama about message ordering. It rejects
// `role: 'tool'` messages that are not immediately preceded by an assistant message
// with matching `tool_calls` (e.g. a tool result at the start of the conversation
// right after `role: 'system'`). Convert those orphan tool results into user
// messages so the conversation stays valid without losing the information.
const sanitizeOllamaCloudMessages = (messages: any[]): any[] => {
	const out: any[] = []
	for (let i = 0; i < messages.length; i++) {
		const m = messages[i]
		if (m.role !== 'tool') {
			out.push(m)
			continue
		}
		const prev = out[out.length - 1]
		const isPrecededByToolCallingAssistant = prev?.role === 'assistant'
			&& Array.isArray(prev.tool_calls)
			&& prev.tool_calls.some((tc: any) => tc.id === m.tool_call_id || tc.id === m.id)
		if (isPrecededByToolCallingAssistant) {
			out.push({
				role: 'tool',
				content: m.content,
				tool_call_id: m.tool_call_id,
			})
		} else {
			voidDevLog(`[sendLLMMessage] ollamaCloud: converting orphan tool result '${m.tool_call_id ?? m.id ?? '?'}' to user message`)
			out.push({
				role: 'user',
				content: typeof m.content === 'string' && m.content.trim().length > 0
					? `Tool result (${m.name ?? 'tool'}): ${m.content}`
					: '(no tool output)',
			})
		}
	}
	return out
}


// ------------ OPENAI-COMPATIBLE ------------


const _sendOpenAICompatibleChat = async ({ messages, onText, onFinalMessage, onError, settingsOfProvider, globalSettings, modelSelectionOptions, modelName: modelName_, _setAborter, providerName, chatMode, separateSystemMessage, overridesOfModel, mcpTools, acpTools, composioTools, allowedTools, allowExternalTools }: SendChatParams_Internal) => {
	const {
		modelName,
		reasoningCapabilities,
		additionalOpenAIPayload,
		specialToolFormat,
	} = getModelCapabilities(providerName, modelName_, overridesOfModel)

	const { providerReasoningIOSettings } = getProviderCapabilities(providerName)

	// reasoning
	const reasoningInfo = getSendableReasoningInfo('Chat', providerName, modelName_, modelSelectionOptions, overridesOfModel) // user's modelName_ here

	// Output token cap. Sent for llama.cpp (whose server defaults to n_predict=-1 / infinite)
	// so a long generation — especially with reasoning enabled — doesn't run to n_ctx
	// exhaustion and get silently truncated by the server's context shift. Only sent when we
	// actually know the model's reserved output space; autodetected/unknown models have none.
	const maxTokens = getReservedOutputTokenSpace(providerName, modelName_, { isReasoningEnabled: !!reasoningInfo?.isReasoningEnabled, overridesOfModel })

	const includeInPayload = {
		...providerReasoningIOSettings?.input?.includeInPayload?.(reasoningInfo),
		...additionalOpenAIPayload
	}

	voidDevLog(`[sendLLMMessage] Reasoning config:`, {
		reasoningCapabilities,
		reasoningInfo,
		includeInPayload: providerReasoningIOSettings?.input?.includeInPayload?.(reasoningInfo),
		finalPayload: includeInPayload
	})

	// tools - only send if model supports native tool calling (specialToolFormat === 'openai-style')
	// Models without specialToolFormat will use XML tool calling instead
	const potentialTools = openAITools(chatMode, mcpTools, acpTools, composioTools, {
		enableMorphFastContext: (modelSelectionOptions?.morphFastContext ?? globalSettings.enableMorphFastContext) && !!globalSettings.morphApiKey,
		enableMediaGeneration: globalSettings.enableMediaGeneration,
		allowedTools,
		allowExternalTools,
		providerName,
	})
	const nativeToolsObj = potentialTools && specialToolFormat === 'openai-style' ?
		{ tools: potentialTools } as const
		: {}

	voidDevLog(`[sendLLMMessage] OpenAI-compatible - chatMode: ${chatMode}, tools count: ${potentialTools?.length ?? 0}, model: ${modelName}, provider: ${providerName}, specialToolFormat: ${specialToolFormat}`)
	if (potentialTools && potentialTools.length > 0 && specialToolFormat === 'openai-style') {
		voidDevLog(`[sendLLMMessage] \u{2705} Sending ${potentialTools.length} tools via native API`)
		voidDevLog(`[sendLLMMessage] Tool names:`, potentialTools.map(t => t.function.name).join(', '))
	} else if (potentialTools && potentialTools.length > 0) {
		voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} NOT sending tools - specialToolFormat is '${specialToolFormat}', will use XML tool calling instead`)
	} else {
		voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} NO TOOLS - chatMode: ${chatMode}, mcpTools: ${mcpTools?.length ?? 0}`)
	}

	// instance
	const openai: OpenAI = await newOpenAICompatibleSDK({ providerName, settingsOfProvider, includeInPayload })
	if (providerName === 'microsoftAzure') {
		// Required to select the model
		(openai as AzureOpenAI).deploymentName = modelName;
	}
	// llama.cpp (Qwen templates): Qwen3/3.5 templates scan backward for a genuine
	// user query and raise if every user message is only a <tool_response>...</tool_response>
	// wrapper. Only inject a synthetic user message when the history actually lacks a
	// real user query, so we don't pollute every tool turn with "Continue." messages
	// that show up in the model's reasoning traces.
	let llamaCppMessages: OpenAILLMChatMessage[] | undefined
	if (providerName === 'llamaCpp') {
		llamaCppMessages = sanitizeOpenAIMessages(messages) as OpenAILLMChatMessage[]
		const hasPlainUserQuery = llamaCppMessages.some(m =>
			m.role === 'user'
			&& typeof m.content === 'string'
			&& (m.content as string).length > 0
			&& !(m.content as string).startsWith('<tool_response>')
		)
		if (!hasPlainUserQuery) {
			voidDevLog(`[sendLLMMessage] llama.cpp: no plain user query found in history; appending synthetic user message`)
			llamaCppMessages.push({ role: 'user', content: ' ' })
		}
	}

	const options: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
		model: modelName,
		// LLMChatMessage is void's provider-agnostic message type; it maps
		// structurally onto the OpenAI wire format but TS can't prove it, so we
		// assert to the exact SDK param type at this boundary.
		messages: (providerName === 'mistral' || providerName === 'openRouter'
			? sanitizeOpenAIMessages(messages)
			: providerName === 'llamaCpp'
				? llamaCppMessages!
				: providerName === 'ollamaCloud'
					? sanitizeOllamaCloudMessages(messages)
					: messages) as unknown as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming['messages'],
		stream: true,
		...nativeToolsObj,
		// Enable parallel tool calls for models that support native tool calling.
		// This tells the model it can return multiple tool calls in a single response.
		// OpenAI default is true, but being explicit ensures Ollama and other
		// OpenAI-compatible providers also receive the signal.
		...(specialToolFormat === 'openai-style' && potentialTools && potentialTools.length > 0 ? { parallel_tool_calls: true } : {}),
		// llama.cpp: explicitly ask the server to parse generated tool calls into structured
		// `tool_calls` deltas instead of leaving them as text in `content` (which we'd then
		// only catch via the XML-tool fallback).
		...(providerName === 'llamaCpp' && specialToolFormat === 'openai-style' && potentialTools && potentialTools.length > 0 ? { parse_tool_calls: true } : {}),
		// llama.cpp: cap output so generation can't run to n_ctx exhaustion (see maxTokens above).
		...(providerName === 'llamaCpp' && typeof maxTokens === 'number' ? { max_tokens: maxTokens } : {}),
		...includeInPayload,
		...additionalOpenAIPayload
		// max_completion_tokens: maxTokens,
	}

	voidDevLog(`[sendLLMMessage] Request options:`, JSON.stringify({
		model: options.model,
		messageCount: options.messages.length,
		hasTools: 'tools' in options,
		toolCount: options.tools?.length ?? 0,
		parallelToolCalls: options.parallel_tool_calls ?? false,
		stream: options.stream
	}))

	// Debug: Log full payload size and check for issues at position 17371
	const fullPayload = JSON.stringify(options)
	voidDevLog(`[sendLLMMessage] Full payload size: ${fullPayload.length} chars`)
	if (fullPayload.length > 17000) {
		voidDevLog(`[sendLLMMessage] Payload around position 17371: "${fullPayload.substring(17350, 17400)}"`)
	}

	const { nameOfFieldInDelta: nameOfReasoningFieldInDelta } = providerReasoningIOSettings?.output ?? {}
	let fullReasoningSoFar = ''
	let fullTextSoFar = ''
	let lastFullTextLength = 0

	const toText = (content: unknown): string => {
		if (!content) return ''
		if (typeof content === 'string') return content
		if (Array.isArray(content)) {
			return content.map(part => {
				if (typeof part === 'string') return part
				if (part && typeof part === 'object' && 'text' in part) return String(part.text ?? '')
				return ''
			}).join('')
		}
		return ''
	}

	let toolCalls: { name: string, id: string, paramsStr: string, thoughtSignature?: string }[] = []
	let finalUsage: { promptTokens: number; completionTokens: number; cachedTokens?: number } | undefined = undefined

	const applyToolCall = (toolCall: any, { isFinal }: { isFinal: boolean }) => {
		if (!toolCall) return

		const index = typeof toolCall.index === 'number' ? toolCall.index : 0
		if (!toolCalls[index]) {
			toolCalls[index] = { name: '', id: '', paramsStr: '' }
		}

		const fn = toolCall.function ?? {}

		if (fn.name && !toolCalls[index].name) {
			toolCalls[index].name = fn.name
			voidDevLog(`[sendLLMMessage] Tool call [${index}] detected: ${fn.name}`)
		}

		if (typeof fn.arguments === 'string') {
			if (isFinal) {
				toolCalls[index].paramsStr = fn.arguments
			} else {
				toolCalls[index].paramsStr += fn.arguments
			}
		}

		if (toolCall.id && !toolCalls[index].id) {
			toolCalls[index].id = toolCall.id
		}

		if (fn.thought_signature || toolCall.thought_signature) {
			toolCalls[index].thoughtSignature = fn.thought_signature || toolCall.thought_signature
		}
	}

	const mapToRawToolCalls = (calls: typeof toolCalls): RawToolCallObj[] => {
		return calls
			.filter(tc => !!tc && !!tc.name) // Skip holes and unnamed tools
			.map(tc => ({
				name: tc.name,
				rawParams: parsePartialJSON(tc.paramsStr),
				isDone: false,
				doneParams: [],
				id: tc.id || generateUuid()
			}))
	}

	voidDevLog(`[sendLLMMessage] Creating request with options:`, JSON.stringify({
		model: options.model,
		messageCount: options.messages?.length,
		hasTools: !!options.tools,
		toolCount: options.tools?.length,
		stream: options.stream
	}))

	// Log the actual messages being sent (first 3 for debugging)
	if (options.messages && options.messages.length > 0) {
		voidDevLog(`[sendLLMMessage] Message sequence (${options.messages.length} messages):`)
		options.messages.forEach((m: any, idx: number) => {
			const contentStr = typeof m.content === 'string' ? m.content : JSON.stringify(m.content)
			const hasTools = !!m.tool_calls
			const toolCount = m.tool_calls?.length ?? 0
			const toolId = m.tool_call_id ?? ''
			voidDevLog(`  [${idx}] role: ${m.role}${toolId ? ` (id: ${toolId})` : ''}${hasTools ? ` (${toolCount} tools)` : ''}, content length: ${contentStr?.length ?? 0}`)
		})

		voidDevLog(`[sendLLMMessage] First message:`, JSON.stringify(options.messages[0], null, 2).substring(0, 500))
		if (options.messages.length > 1) {
			voidDevLog(`[sendLLMMessage] Last message:`, JSON.stringify(options.messages[options.messages.length - 1], null, 2).substring(0, 500))
		}
	}

	// Create an AbortController BEFORE create() so a fast Stop during the
	// connect/request-send window (before .then() resolves and wires the stream's
	// own controller) actually cancels the in-flight fetch instead of being a no-op.
	const preConnectAbort = new AbortController()
	_setAborter(() => preConnectAbort.abort())

	openai.chat.completions
		.create(options, { signal: preConnectAbort.signal })
		.then(async response => {
			voidDevLog(`[sendLLMMessage] Request created successfully, starting to read stream`)
			// Switch the aborter to the stream's own controller for in-stream cancels.
			_setAborter(() => response.controller.abort())

			// Import XML stripping function for streaming
			const { stripXMLBlocks } = await import('../../common/helpers/extractXMLTools.js')

			// when receive text
			let chunkCount = 0
			let lastFinishReason: string | undefined
			for await (const chunk of response) {
				chunkCount++
				if (chunkCount <= 3) {
					voidDevLog(`[sendLLMMessage] Chunk ${chunkCount} received:`, JSON.stringify(chunk, null, 2))
				}
				const choice = chunk.choices?.[0]
				if (!choice) {
					voidDevLog(`[sendLLMMessage] Chunk ${chunkCount} has no choice, skipping`)
					continue
				}

				if (choice.finish_reason) {
					lastFinishReason = choice.finish_reason
				}

				if (choice.finish_reason && choice.finish_reason !== 'stop' && choice.finish_reason !== 'tool_calls') {
					// 'length' means the response hit the max-output-token cap mid-generation.
					// The partial text has already been streamed via onText, so treat this as
					// successful completion (fall through to onFinalMessage below) and surface a
					// non-blocking warning instead of discarding the partial content as an error.
					if (choice.finish_reason === 'length') {
						voidDevWarn(`[sendLLMMessage] Response truncated at max output tokens (finish_reason: "length"); delivering partial content.`)
					}
					else {
						voidDevLog(`[sendLLMMessage] Unexpected finish_reason: ${choice.finish_reason}, chunk:`, JSON.stringify(chunk))
						onError({ message: `Model ended response with finish_reason "${choice.finish_reason}"`, fullError: null })
						return
					}
				}

				// Capture usage statistics if available
				if (chunk.usage) {
					// prompt_tokens_details.cached_tokens is populated by OpenAI-compatible
					// servers with prompt caching (e.g. llama.cpp's cache_n). Surfaced for the
					// UI / token calibration rather than discarded.
					const cachedTokens = chunk.usage.prompt_tokens_details?.cached_tokens
					finalUsage = {
						promptTokens: chunk.usage.prompt_tokens,
						completionTokens: chunk.usage.completion_tokens,
						...(typeof cachedTokens === 'number' ? { cachedTokens } : {}),
					};
					voidDevLog(`[sendLLMMessage] Usage stats received:`, finalUsage);
				}

				const delta = choice.delta ?? {}
				const newText = toText(delta.content)
				fullTextSoFar += newText

				for (const toolDelta of delta.tool_calls ?? []) {
					applyToolCall(toolDelta, { isFinal: false })
				}
				if (choice.finish_reason === 'tool_calls') {
					for (let i = 0; i < toolCalls.length; i++) {
						if (!toolCalls[i]) continue // Guard against sparse array from non-contiguous tool indices
						applyToolCall({ function: { name: toolCalls[i].name, arguments: toolCalls[i].paramsStr }, id: toolCalls[i].id, index: i }, { isFinal: true })
					}
				}

				if (nameOfReasoningFieldInDelta) {
					// Check configured field first, then fallback to common alternatives
					// Different providers use different field names for reasoning content
					const reasoningDelta = readReasoningFromDelta(delta, nameOfReasoningFieldInDelta)
					if (reasoningDelta && chunkCount <= 5) {
						voidDevLog(`[sendLLMMessage] Chunk ${chunkCount} reasoning delta:`, reasoningDelta.substring(0, 100))
					}
					// Accumulate reasoning if reasoning is enabled in settings, OR if it's an unrecognized model (reasoningInfo is null)
					if (!reasoningInfo || reasoningInfo.isReasoningEnabled) {
						fullReasoningSoFar += reasoningDelta
					}
				}

				// Strip XML blocks from text during streaming if model doesn't support native tools
				const displayText = !specialToolFormat ? stripXMLBlocks(fullTextSoFar) : fullTextSoFar

				onText({
					fullText: displayText,
					fullReasoning: fullReasoningSoFar,
					textDelta: newText,
					reasoningDelta: nameOfReasoningFieldInDelta ? readReasoningFromDelta(delta, nameOfReasoningFieldInDelta) : undefined,
					toolCalls: toolCalls.length === 0 ? undefined : mapToRawToolCalls(toolCalls),
					// Pass raw text so chatThreadService can detect XML tool calls for repetition detection
					_rawTextBeforeStripping: !specialToolFormat ? fullTextSoFar : undefined,
				})
			}
			// on final
			const truncatedFullText = fullTextSoFar.length > 500 ? fullTextSoFar.substring(0, 500) + '...' : fullTextSoFar;
			const truncatedReasoning = fullReasoningSoFar.length > 500 ? fullReasoningSoFar.substring(0, 500) + '...' : fullReasoningSoFar;
			
			voidDevLog(`[sendLLMMessage] Stream completed. Total chunks: ${chunkCount}, fullText: "${truncatedFullText}", reasoning: "${truncatedReasoning}", toolCalls count: ${toolCalls.length}`)

			// Fallback: If no native tool call detected, check for XML tool calls in both content and reasoning
			if (toolCalls.length === 0 && (fullTextSoFar || fullReasoningSoFar)) {
				const { extractXMLToolCalls, stripXMLBlocks } = await import('../../common/helpers/extractXMLTools.js')
				
				// 1. Check main content
				const xmlToolCallsInText = extractXMLToolCalls(fullTextSoFar)
				for (let i = 0; i < xmlToolCallsInText.length; i++) {
					const call = xmlToolCallsInText[i]
					toolCalls.push({
						name: call.toolName,
						paramsStr: JSON.stringify(call.parameters),
						id: `xml-tool-call-text-${i}`
					})
				}
				if (xmlToolCallsInText.length > 0) {
					fullTextSoFar = stripXMLBlocks(fullTextSoFar)
					voidDevLog(`[sendLLMMessage] \u{2705} Extracted ${xmlToolCallsInText.length} XML tool calls from text`)
				}
				
				// 2. Check reasoning (Nemotron and other models sometimes put tools here)
				if (toolCalls.length === 0 && fullReasoningSoFar) {
					const xmlToolCallsInReasoning = extractXMLToolCalls(fullReasoningSoFar)
					for (let i = 0; i < xmlToolCallsInReasoning.length; i++) {
						const call = xmlToolCallsInReasoning[i]
						toolCalls.push({
							name: call.toolName,
							paramsStr: JSON.stringify(call.parameters),
							id: `xml-tool-call-reasoning-${i}`
						})
					}
					if (xmlToolCallsInReasoning.length > 0) {
						fullReasoningSoFar = stripXMLBlocks(fullReasoningSoFar)
						voidDevLog(`[sendLLMMessage] \u{2705} Extracted ${xmlToolCallsInReasoning.length} XML tool calls from reasoning`)
					}
				}
			}

			// Enhanced empty response detection for Ollama
			const hasEmptyResponse = !fullTextSoFar && !fullReasoningSoFar && toolCalls.length === 0

			if (hasEmptyResponse) {
				voidDevLog(`[sendLLMMessage] \u{274C} Empty response detected`)
				// ... (guiding messages for Ollama)
				onError({ message: 'A-Coder IDE: Response from model was empty.', fullError: null })
			}
			else {
				const finalToolCalls = toolCalls
					.filter(tc => !!tc && !!tc.name)
					.map(tc => rawToolCallObjOfParamsStr(tc.name, tc.paramsStr, tc.id, tc.thoughtSignature))
					.filter(tc => !!tc) as RawToolCallObj[]
				const toolCallObj = finalToolCalls.length > 0 ? { toolCalls: finalToolCalls } : {}
				
				const firstValidToolCall = toolCalls.find(tc => !!tc);
				const anthropicReasoning: AnthropicReasoning[] | null = fullReasoningSoFar ? [{ type: 'thinking', thinking: fullReasoningSoFar, signature: firstValidToolCall?.thoughtSignature || '' }] : null
				
				voidDevLog(`[sendLLMMessage] Final message - text length: ${fullTextSoFar.length}, reasoning length: ${fullReasoningSoFar.length}, toolCalls: ${finalToolCalls.length}`)
				onFinalMessage({ fullText: fullTextSoFar, fullReasoning: fullReasoningSoFar, anthropicReasoning, ...toolCallObj, usage: finalUsage, stopReason: lastFinishReason });
			}
		})
		// when error/fail - this catches errors of both .create() and .then(for await)
		.catch(async error => {
			voidDevLog(`[sendLLMMessage] Error caught:`, error)
			voidDevLog(`[sendLLMMessage] Error type:`, error?.constructor?.name)
			voidDevLog(`[sendLLMMessage] Error status:`, error?.status)
			voidDevLog(`[sendLLMMessage] Error message:`, error?.message)

			// Retry on 500 errors (server-side issues) - wait 3 seconds and try once more
			// Don't call onError yet - let the UI keep showing "thinking" state
			if (error instanceof OpenAI.APIError && error.status === 500 && !('_isRetry' in options && options._isRetry)) {
				voidDevLog(`[sendLLMMessage] \u{23F3} Server returned 500 error, retrying in 3 seconds...`)
				await new Promise(resolve => setTimeout(resolve, 3000))
				voidDevLog(`[sendLLMMessage] \u{1F504} Retrying request...`)

				// Retry the request with a flag to prevent infinite retries
				const retryOptions: typeof options & { _isRetry: boolean } = { ...options, _isRetry: true }
				// Fresh controller so a Stop during the 3s wait / retry connect still cancels.
				const retryAbort = new AbortController()
				_setAborter(() => retryAbort.abort())
				openai.chat.completions
					.create(retryOptions, { signal: retryAbort.signal })
					.then(async retryStreamOrResponse => {
						voidDevLog(`[sendLLMMessage] \u{2705} Retry succeeded, processing response`)
						// The SDK types .create() conservatively (the stream flag is only
						// known at runtime here), so assert to the streaming shape we know
						// we requested: an async iterable of chunks that also exposes the
						// AbortController (present on SDK Stream objects).
						const retryStream = retryStreamOrResponse as unknown as AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> & { controller?: AbortController }
						_setAborter(() => retryStream.controller?.abort?.())

						// Reset state for retry
						fullTextSoFar = ''
						fullReasoningSoFar = ''
						toolCalls = []
						lastFullTextLength = 0

						const { stripXMLBlocks } = await import('../../common/helpers/extractXMLTools.js')

						let chunkCount = 0
						let retryFinishReason: string | undefined
						for await (const chunk of retryStream) {
							chunkCount++
							const choice = chunk.choices?.[0]
							if (!choice) continue

							if (choice.finish_reason) {
								retryFinishReason = choice.finish_reason
							}

							if (choice.finish_reason && choice.finish_reason !== 'stop' && choice.finish_reason !== 'tool_calls') {
								onError({ message: `Model ended response with finish_reason "${choice.finish_reason}"`, fullError: null })
								return
							}

							const delta = choice.delta ?? {}
							fullTextSoFar += toText(delta.content)

							for (const toolDelta of delta.tool_calls ?? []) {
								applyToolCall(toolDelta, { isFinal: false })
							}
							if (choice.finish_reason === 'tool_calls') {
								for (let i = 0; i < toolCalls.length; i++) {
									if (!toolCalls[i]) continue // Guard against sparse array from non-contiguous tool indices
									applyToolCall({ function: { name: toolCalls[i].name, arguments: toolCalls[i].paramsStr }, id: toolCalls[i].id, index: i }, { isFinal: true })
								}
							}

							if (nameOfReasoningFieldInDelta && (!reasoningInfo || reasoningInfo.isReasoningEnabled)) {
								const reasoningDelta = readReasoningFromDelta(delta, nameOfReasoningFieldInDelta)
								fullReasoningSoFar += reasoningDelta
							}

							const displayText = !specialToolFormat ? stripXMLBlocks(fullTextSoFar) : fullTextSoFar

							// Calculate textDelta for this chunk
							const newText = fullTextSoFar.slice(lastFullTextLength)
							lastFullTextLength = fullTextSoFar.length

							onText({
								fullText: displayText,
								fullReasoning: fullReasoningSoFar,
								textDelta: newText,
								reasoningDelta: nameOfReasoningFieldInDelta && (!reasoningInfo || reasoningInfo.isReasoningEnabled) ? readReasoningFromDelta(delta, nameOfReasoningFieldInDelta) : undefined,
								toolCalls: toolCalls.length === 0 ? undefined : mapToRawToolCalls(toolCalls),
								_rawTextBeforeStripping: !specialToolFormat ? fullTextSoFar : undefined,
							})
						}

						voidDevLog(`[sendLLMMessage] Retry stream completed. Chunks: ${chunkCount}`)

						// Fallback: If no native tool call detected, check for XML tool calls in both content and reasoning
						if (toolCalls.length === 0 && (fullTextSoFar || fullReasoningSoFar)) {
							const { extractXMLToolCalls, stripXMLBlocks } = await import('../../common/helpers/extractXMLTools.js')
							
							const xmlToolCallsInText = extractXMLToolCalls(fullTextSoFar)
							for (let i = 0; i < xmlToolCallsInText.length; i++) {
								const call = xmlToolCallsInText[i]
								toolCalls.push({
									name: call.toolName,
									paramsStr: JSON.stringify(call.parameters),
									id: `xml-tool-call-text-retry-${i}`
								})
							}
							if (xmlToolCallsInText.length > 0) {
								fullTextSoFar = stripXMLBlocks(fullTextSoFar)
							}
							
							if (toolCalls.length === 0 && fullReasoningSoFar) {
								const xmlToolCallsInReasoning = extractXMLToolCalls(fullReasoningSoFar)
								for (let i = 0; i < xmlToolCallsInReasoning.length; i++) {
									const call = xmlToolCallsInReasoning[i]
									toolCalls.push({
										name: call.toolName,
										paramsStr: JSON.stringify(call.parameters),
										id: `xml-tool-call-reasoning-retry-${i}`
									})
								}
								if (xmlToolCallsInReasoning.length > 0) {
									fullReasoningSoFar = stripXMLBlocks(fullReasoningSoFar)
								}
							}
						}

						const finalToolCalls = toolCalls
							.filter(tc => !!tc && !!tc.name)
							.map(tc => rawToolCallObjOfParamsStr(tc.name, tc.paramsStr, tc.id, tc.thoughtSignature))
							.filter(tc => !!tc) as RawToolCallObj[]
						const toolCallObj = finalToolCalls.length > 0 ? { toolCalls: finalToolCalls } : {}
						
						const firstValidToolCall = toolCalls.find(tc => !!tc);
						const anthropicReasoning: AnthropicReasoning[] | null = fullReasoningSoFar ? [{ type: 'thinking', thinking: fullReasoningSoFar, signature: firstValidToolCall?.thoughtSignature || '' }] : null
						onFinalMessage({ fullText: fullTextSoFar, fullReasoning: fullReasoningSoFar, anthropicReasoning, ...toolCallObj, stopReason: retryFinishReason })
					})
					.catch(retryError => {
						voidDevLog(`[sendLLMMessage] \u{274C} Retry also failed:`, retryError?.message)
						try {
							onError({ message: retryError + '', fullError: retryError })
						} catch (e) {
							console.error('[sendLLMMessage] onError threw in retry handler:', e)
						}
					})
				return
			}

			if (error instanceof OpenAI.APIError && error.status === 401) { onError({ message: invalidApiKeyMessage(providerName), fullError: error }); }
			else { onError({ message: error + '', fullError: error }); }
		})
}



type OpenAIModel = {
	id: string;
	created: number;
	object: 'model';
	owned_by: string;
	// llama.cpp includes the server's loaded context size in model metadata.
	meta?: { n_ctx?: number };
}
const _openaiCompatibleList = async ({ onSuccess: onSuccess_, onError: onError_, settingsOfProvider, providerName }: ListParams_Internal<OpenAIModel>) => {
	const onSuccess = ({ models }: { models: OpenAIModel[] }) => {
		onSuccess_({ models })
	}
	const onError = ({ error }: { error: string }) => {
		onError_({ error })
	}
	try {
		const openai = await newOpenAICompatibleSDK({ providerName, settingsOfProvider })
		openai.models.list()
			.then(async (response) => {
				const models: OpenAIModel[] = []
				models.push(...response.data)
				while (response.hasNextPage()) {
					models.push(...(await response.getNextPage()).data)
				}
				// LM Studio's /v1/models response omits context length, but its native
				// /api/v0/models endpoint exposes max_context_length. Fetch it in parallel
				// and merge the result into the model objects we return.
				if (providerName === 'lmStudio') {
					try {
						const endpoint = (settingsOfProvider.lmStudio.endpoint || '').replace(/\/+$/, '')
						if (endpoint) {
							const lmStudioRes = await fetch(`${endpoint}/api/v0/models`)
							if (lmStudioRes.ok) {
								const lmStudioData: { data?: LMStudioModel[] } = await lmStudioRes.json()
								const contextById = new Map(
									(lmStudioData.data ?? [])
										.map(m => [m.id, m.max_context_length] as const)
										.filter((entry): entry is readonly [string, number] => typeof entry[1] === 'number' && entry[1] > 0)
								)
								for (const model of models) {
									const ctx = contextById.get(model.id)
									if (ctx !== undefined) {
										model.meta = { ...(model.meta ?? {}), n_ctx: ctx }
									}
								}
							}
						}
					} catch {
						// Don't let the secondary context fetch break model listing.
					}
				}
				onSuccess({ models })
			})
			.catch((error) => {
				onError({ error: error + '' })
			})
	}
	catch (error) {
		onError({ error: error + '' })
	}
}

type LMStudioModel = {
	id: string;
	max_context_length?: number;
}

// Ollama Cloud exposes model tags via the Ollama /api/tags endpoint and context length
// via /api/show (model_info.<arch>.context_length). We use these to autodetect context.
type OllamaCloudTagModel = {
	name: string;
	model: string;
	modified_at: string;
	size: number;
	digest: string;
	details: {
		parent_model: string;
		format: string;
		family: string;
		families: string[] | null;
		parameter_size: string;
		quantization_level: string;
	};
	capabilities?: string[];
}
type OllamaCloudShowResponse = {
	model_info?: Record<string, number>;
}

const _fetchOllamaShowContext = async (endpoint: string, modelName: string): Promise<number | undefined> => {
	const res = await fetch(`${endpoint}/api/show`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ name: modelName }),
	})
	if (!res.ok) return undefined
	const data: OllamaCloudShowResponse = await res.json()
	if (!data.model_info) return undefined
	for (const key in data.model_info) {
		if (key.endsWith('.context_length')) {
			const val = data.model_info[key]
			if (typeof val === 'number' && val > 0) return val
		}
	}
	return undefined
}

const _ollamaCloudList = async ({ onSuccess: onSuccess_, onError: onError_, settingsOfProvider }: ListParams_Internal<OpenAIModel>) => {
	const onSuccess = ({ models }: { models: OpenAIModel[] }) => {
		onSuccess_({ models })
	}
	const onError = ({ error }: { error: string }) => {
		onError_({ error })
	}
	try {
		const endpoint = (settingsOfProvider.ollamaCloud.endpoint || '').replace(/\/+$/, '')
		if (!endpoint) {
			onError({ error: 'Ollama Cloud endpoint was empty.' })
			return
		}
		const res = await fetch(`${endpoint}/api/tags`)
		if (!res.ok) {
			onError({ error: `Ollama Cloud /api/tags failed (${res.status} ${res.statusText})` })
			return
		}
		const data: { models?: OllamaCloudTagModel[] } = await res.json()
		const tagModels = data.models ?? []
		const models: OpenAIModel[] = []
		for (const tagModel of tagModels) {
			const n_ctx = await _fetchOllamaShowContext(endpoint, tagModel.name)
			models.push({
				id: tagModel.name,
				created: Math.floor(new Date(tagModel.modified_at).getTime() / 1000),
				object: 'model',
				owned_by: 'ollama',
				...(n_ctx !== undefined ? { meta: { n_ctx } } : {}),
			})
		}
		onSuccess({ models })
	}
	catch (error) {
		onError({ error: error + '' })
	}
}




// ------------ ANTHROPIC (HELPERS) ------------
const toAnthropicTool = (toolInfo: InternalToolInfo) => {
	const { name, description, params } = toolInfo
	// Anthropic path: keep the historical all-string schema. Take only `description` +
	// `type: 'string'` explicitly (no spread) so MCP tools' richer JSON-Schema fields
	// (enum/items) don't leak in alongside a forced string type now that InternalToolInfo
	// carries them. The OpenAI-compatible path emits the real schema; see toOpenAICompatibleTool.
	const paramsWithType: { [s: string]: { description: string; type: 'string' } } = {}
	for (const key in params) { paramsWithType[key] = { description: params[key].description, type: 'string' } }
	return {
		name: name,
		description: description,
		input_schema: {
			type: 'object',
			properties: paramsWithType,
			// required: Object.keys(params),
		},
	} satisfies Anthropic.Messages.Tool
}

const anthropicTools = (chatMode: ChatMode | null, mcpTools: InternalToolInfo[] | undefined, acpTools: InternalToolInfo[] | undefined, composioTools: InternalToolInfo[] | undefined, options?: { enableMorphFastContext?: boolean; enableMediaGeneration?: boolean; allowedTools?: string[]; allowExternalTools?: boolean }) => {
	const allowedTools = availableTools(chatMode, mcpTools, composioTools, acpTools, options)
	if (!allowedTools || Object.keys(allowedTools).length === 0) return null

	const anthropicTools: Anthropic.Messages.ToolUnion[] = []
	for (const t in allowedTools ?? {}) {
		anthropicTools.push(toAnthropicTool(allowedTools[t]))
	}
	return anthropicTools
}



// ------------ ANTHROPIC ------------
const sendAnthropicChat = async ({ messages, providerName, onText, onFinalMessage, onError, settingsOfProvider, globalSettings, modelSelectionOptions, overridesOfModel, modelName: modelName_, _setAborter, separateSystemMessage, chatMode, mcpTools, acpTools, composioTools, allowedTools, allowExternalTools }: SendChatParams_Internal) => {
	const {
		modelName,
		specialToolFormat,
	} = getModelCapabilities(providerName, modelName_, overridesOfModel)

	const thisConfig = settingsOfProvider.anthropic
	const { providerReasoningIOSettings } = getProviderCapabilities(providerName)

	// reasoning
	const reasoningInfo = getSendableReasoningInfo('Chat', providerName, modelName_, modelSelectionOptions, overridesOfModel) // user's modelName_ here
	const includeInPayload = providerReasoningIOSettings?.input?.includeInPayload?.(reasoningInfo) || {}

	// anthropic-specific - max tokens
	const maxTokens = getReservedOutputTokenSpace(providerName, modelName_, { isReasoningEnabled: !!reasoningInfo?.isReasoningEnabled, overridesOfModel })

	// tools
	const potentialTools = anthropicTools(chatMode, mcpTools, acpTools, composioTools, {
		enableMorphFastContext: (modelSelectionOptions?.morphFastContext ?? globalSettings.enableMorphFastContext) && !!globalSettings.morphApiKey,
		enableMediaGeneration: globalSettings.enableMediaGeneration,
		allowedTools,
		allowExternalTools,
	})
	const nativeToolsObj = potentialTools && specialToolFormat === 'anthropic-style' ?
		{ tools: potentialTools, tool_choice: { type: 'auto' } } as const
		: {}

	voidDevLog(`[sendLLMMessage] Anthropic - chatMode: ${chatMode}, tools count: ${potentialTools?.length ?? 0}, specialToolFormat: ${specialToolFormat}`)
	if (potentialTools && potentialTools.length > 0 && specialToolFormat === 'anthropic-style') {
		voidDevLog(`[sendLLMMessage] Tool names:`, potentialTools.map(t => t.name).join(', '))
	} else if (potentialTools && potentialTools.length > 0) {
		voidDevLog(`[sendLLMMessage] \u{26A0}\u{FE0F} TOOLS NOT SENT - specialToolFormat is ${specialToolFormat}, expected 'anthropic-style'`)
	}

	// instance
	const anthropic = new Anthropic({
		apiKey: thisConfig.apiKey,
		dangerouslyAllowBrowser: true
	});

	const stream = anthropic.messages.stream({
		system: separateSystemMessage ?? undefined,
		messages: messages as AnthropicLLMChatMessage[],
		model: modelName,
		max_tokens: maxTokens ?? 4_096, // anthropic requires this
		...includeInPayload,
		...nativeToolsObj,
	})

	// when receive text
	let fullText = ''
	let fullReasoning = ''

	let toolCallsAccumulator: { name: string, id: string, paramsStr: string }[] = []


	const runOnText = () => {
		onText({
			fullText,
			fullReasoning,
			toolCalls: toolCallsAccumulator.length === 0 ? undefined : toolCallsAccumulator
				.filter(tc => !!tc)
				.map(tc => ({
					name: tc.name,
					rawParams: parsePartialJSON(tc.paramsStr),
					isDone: false,
					doneParams: [],
					id: tc.id || generateUuid()
				})),
			// Pass raw text for repetition detection (Anthropic uses native tools, so no XML stripping)
			_rawTextBeforeStripping: fullText,
		})
	}
	// there are no events for tool_use, it comes in at the end
	stream.on('streamEvent', e => {
		// start block
		if (e.type === 'content_block_start') {
			if (e.content_block.type === 'text') {
				if (fullText) fullText += '\n\n' // starting a 2nd text block
				fullText += e.content_block.text
				runOnText()
			}
			else if (e.content_block.type === 'thinking') {
				if (fullReasoning) fullReasoning += '\n\n' // starting a 2nd reasoning block
				fullReasoning += e.content_block.thinking
				runOnText()
			}
			else if (e.content_block.type === 'redacted_thinking') {
				voidDevLog('delta', e.content_block.type)
				if (fullReasoning) fullReasoning += '\n\n' // starting a 2nd reasoning block
				fullReasoning += '[redacted_thinking]'
				runOnText()
			}
			else if (e.content_block.type === 'tool_use') {
				const index = e.index
				toolCallsAccumulator[index] = { name: e.content_block.name ?? '', id: e.content_block.id, paramsStr: '' }
				runOnText()
			}
		}

		// delta
		else if (e.type === 'content_block_delta') {
			if (e.delta.type === 'text_delta') {
				fullText += e.delta.text
				onText({
					fullText,
					fullReasoning,
					textDelta: e.delta.text,
					toolCalls: toolCallsAccumulator.length === 0 ? undefined : toolCallsAccumulator
						.filter(tc => !!tc)
						.map(tc => ({
							name: tc.name,
							rawParams: parsePartialJSON(tc.paramsStr),
							isDone: false,
							doneParams: [],
							id: tc.id || generateUuid()
						})),
					// Pass raw text for repetition detection (Anthropic uses native tools, so no XML stripping)
					_rawTextBeforeStripping: fullText,
				})
			}
			else if (e.delta.type === 'thinking_delta') {
				fullReasoning += e.delta.thinking
				onText({
					fullText,
					fullReasoning,
					reasoningDelta: e.delta.thinking,
					toolCalls: toolCallsAccumulator.length === 0 ? undefined : toolCallsAccumulator
						.filter(tc => !!tc)
						.map(tc => ({
							name: tc.name,
							rawParams: parsePartialJSON(tc.paramsStr),
							isDone: false,
							doneParams: [],
							id: tc.id || generateUuid()
						})),
					// Pass raw text for repetition detection (Anthropic uses native tools, so no XML stripping)
					_rawTextBeforeStripping: fullText,
				})
			}
			else if (e.delta.type === 'input_json_delta') { // tool use
				const index = e.index
				if (toolCallsAccumulator[index]) {
					toolCallsAccumulator[index].paramsStr += e.delta.partial_json ?? ''
				}
				runOnText()
			}
		}
	})

	// on done - (or when error/fail) - this is called AFTER last streamEvent
	stream.on('finalMessage', (response) => {
		const anthropicReasoning = response.content.filter(c => c.type === 'thinking' || c.type === 'redacted_thinking')
		const tools = response.content.filter(c => c.type === 'tool_use') as Anthropic.Messages.ToolUseBlock[]
		const finalToolCalls = tools.map(t => rawToolCallObjOfAnthropicParams(t)).filter(tc => !!tc) as RawToolCallObj[]
		const toolCallObj = finalToolCalls.length > 0 ? { toolCalls: finalToolCalls } : {}

		// Capture real token usage so callers can anchor context-size accounting
		// on it instead of re-estimating the entire prompt. Anthropic's
		// `input_tokens` EXCLUDES cache hits, so add cache reads/creations back in
		// to get the true prompt size (what matters for context-window math).
		const u = response.usage
		const usage = u ? {
			promptTokens: (u.input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0),
			completionTokens: u.output_tokens ?? 0,
			...(typeof u.cache_read_input_tokens === 'number' ? { cachedTokens: u.cache_read_input_tokens } : {}),
		} : undefined

		onFinalMessage({ fullText, fullReasoning, anthropicReasoning, ...toolCallObj, usage, stopReason: response.stop_reason ?? undefined })
	})
	// on error
	stream.on('error', (error) => {
		if (error instanceof Anthropic.APIError && error.status === 401) { onError({ message: invalidApiKeyMessage(providerName), fullError: error }) }
		else { onError({ message: error + '', fullError: error }) }
	})
	_setAborter(() => stream.controller.abort())
}



// ------------ MISTRAL ------------
// https://docs.mistral.ai/api/#tag/fim
const sendMistralFIM = ({ messages, onFinalMessage, onError, settingsOfProvider, globalSettings, overridesOfModel, modelName: modelName_, _setAborter, providerName }: SendFIMParams_Internal) => {
	const { modelName, supportsFIM } = getModelCapabilities(providerName, modelName_, overridesOfModel)
	if (!supportsFIM) {
		if (modelName === modelName_)
			onError({ message: `Model ${modelName} does not support FIM.`, fullError: null })
		else
			onError({ message: `Model ${modelName_} (${modelName}) does not support FIM.`, fullError: null })
		return
	}

	const mistral = new MistralCore({ apiKey: settingsOfProvider.mistral.apiKey })
	fimComplete(mistral,
		{
			model: modelName,
			prompt: messages.prefix,
			suffix: messages.suffix,
			stream: false,
			maxTokens: 300,
			stop: messages.stopTokens,
		})
		.then(async response => {

			// unfortunately, _setAborter() does not exist
			let content = response?.ok ? response.value.choices?.[0]?.message?.content ?? '' : '';
			const fullText = typeof content === 'string' ? content
				: content.map(chunk => (chunk.type === 'text' ? chunk.text : '')).join('')

			onFinalMessage({ fullText, fullReasoning: '', anthropicReasoning: null });
		})
		.catch(error => {
			onError({ message: error + '', fullError: error });
		})
}


// ------------ OLLAMA ------------
const newOllamaSDK = ({ endpoint }: { endpoint: string }) => {
	// if endpoint is empty, normally ollama will send to 11434, but we want it to fail - the user should type it in
	if (!endpoint) throw new Error(`Ollama Endpoint was empty (please enter ${defaultProviderSettings.ollama.endpoint} in A-Coder IDE if you want the default url).`)
	const ollama = new Ollama({ host: endpoint })
	return ollama
}

const ollamaList = async ({ onSuccess: onSuccess_, onError: onError_, settingsOfProvider }: ListParams_Internal<OllamaModelResponse>) => {
	const onSuccess = ({ models }: { models: OllamaModelResponse[] }) => {
		onSuccess_({ models })
	}
	const onError = ({ error }: { error: string }) => {
		onError_({ error })
	}
	try {
		const thisConfig = settingsOfProvider.ollama
		const endpoint = (thisConfig.endpoint || '').replace(/\/+$/, '')
		const ollama = newOllamaSDK({ endpoint: thisConfig.endpoint })
		const response = await ollama.list()
		const { models } = response

		// Two-step flow (mirrors the Ollama Cloud path): /api/tags gives model names,
		// /api/show gives context_length in model_info.<arch>.context_length. We reuse
		// the same raw-fetch helper rather than the SDK's show() so local and cloud
		// autodetect context windows identically. /api/show is a lightweight metadata
		// lookup, so fire all calls in parallel.
		await Promise.all(models.map(async (model) => {
			try {
				const ctxLen = await _fetchOllamaShowContext(endpoint, model.name)
				if (typeof ctxLen === 'number' && ctxLen > 0 && model.details) {
					// Ollama's ModelDetails type doesn't declare context_length (the
					// field comes from /api/show's model_info, not /api/tags), so
					// extend the type instead of reaching through `any`.
					;(model.details as Omit<typeof model.details, never> & { context_length?: number }).context_length = ctxLen
				}
			} catch {
				// Silently skip models that fail /api/show (e.g. running model with no Modelfile)
			}
		}))

		onSuccess({ models })
	}
	catch (error) {
		onError({ error: error + '' })
	}
}

const sendOllamaFIM = ({ messages, onFinalMessage, onError, settingsOfProvider, modelName, _setAborter }: SendFIMParams_Internal) => {
	const thisConfig = settingsOfProvider.ollama
	const ollama = newOllamaSDK({ endpoint: thisConfig.endpoint })

	let fullText = ''
	ollama.generate({
		model: modelName,
		prompt: messages.prefix,
		suffix: messages.suffix,
		options: {
			stop: messages.stopTokens,
			num_predict: 300, // max tokens
			// repeat_penalty: 1,
		},
		raw: true,
		stream: true, // stream is not necessary but lets us expose the
	})
		.then(async stream => {
			_setAborter(() => stream.abort())
			for await (const chunk of stream) {
				const newText = chunk.response
				fullText += newText
			}
			onFinalMessage({ fullText, fullReasoning: '', anthropicReasoning: null })
		})
		// when error/fail
		.catch((error) => {
			onError({ message: error + '', fullError: error })
		})
}

// ---------------- GEMINI NATIVE IMPLEMENTATION ----------------

const toGeminiFunctionDecl = (toolInfo: InternalToolInfo) => {
	const { name, description, params } = toolInfo
	return {
		name,
		description,
		parameters: {
			type: Type.OBJECT,
			properties: Object.entries(params).reduce((acc, [key, value]) => {
				acc[key] = {
					type: Type.STRING,
					description: value.description
				};
				return acc;
			}, {} as Record<string, Schema>)
		}
	} satisfies FunctionDeclaration
}

const geminiTools = (chatMode: ChatMode | null, mcpTools: InternalToolInfo[] | undefined, acpTools: InternalToolInfo[] | undefined, composioTools: InternalToolInfo[] | undefined, options?: { enableMorphFastContext?: boolean; enableMediaGeneration?: boolean; allowedTools?: string[]; allowExternalTools?: boolean }): GeminiTool[] | null => {
	const allowedTools = availableTools(chatMode, mcpTools, composioTools, acpTools, options)
	if (!allowedTools || Object.keys(allowedTools).length === 0) return null
	const functionDecls: FunctionDeclaration[] = []
	for (const t in allowedTools ?? {}) {
		functionDecls.push(toGeminiFunctionDecl(allowedTools[t]))
	}
	const tools: GeminiTool = { functionDeclarations: functionDecls, }
	return [tools]
}



// Enhanced Ollama chat with fallback for better tool calling reliability
const _sendOllamaChatWithFallback = async (params: SendChatParams_Internal) => {
	const { chatMode, mcpTools, acpTools, composioTools, modelName, globalSettings, modelSelectionOptions, allowedTools, allowExternalTools } = params

	// Check if model supports native tool calling
	const { specialToolFormat } = getModelCapabilities('ollama', modelName, params.overridesOfModel)
	const hasNativeTools = specialToolFormat === 'openai-style'
	const potentialTools = openAITools(chatMode, mcpTools, acpTools, composioTools, {
		enableMorphFastContext: (modelSelectionOptions?.morphFastContext ?? globalSettings.enableMorphFastContext) && !!globalSettings.morphApiKey,
		enableMediaGeneration: globalSettings.enableMediaGeneration,
		allowedTools,
		allowExternalTools,
		providerName: 'ollama',
	})
	const hasTools = potentialTools && potentialTools.length > 0

	voidDevLog(`[sendOllamaChatWithFallback] Model: ${modelName}, specialToolFormat: ${specialToolFormat}, hasTools: ${hasTools}`)

	// _sendOpenAICompatibleChat fires its streaming chain (create().then().catch())
	// and resolves before the stream actually runs, so a plain try/catch around it
	// can never see stream/API failures. Wrap onFinalMessage/onError so the promise
	// below settles only when the stream truly completes (success or error).
	const runAttempt = (attemptParams: SendChatParams_Internal): Promise<{ failed: boolean; error: any }> => {
		return new Promise(resolve => {
			let settled = false
			const settle = (result: { failed: boolean; error: any }) => {
				if (!settled) { settled = true; resolve(result) }
			}
			const wrappedParams: SendChatParams_Internal = {
				...attemptParams,
				onFinalMessage: (p) => { attemptParams.onFinalMessage(p); settle({ failed: false, error: null }) },
				onError: (p) => { attemptParams.onError(p); settle({ failed: true, error: p.fullError ?? null }) },
			}
			// .catch covers synchronous setup errors thrown before the callbacks are wired.
			_sendOpenAICompatibleChat(wrappedParams).catch(e => settle({ failed: true, error: e }))
		})
	}
	const isTransientServerError = (error: any) => error?.status >= 500 && error?.status < 600

	// For models with native tool support, try OpenAI-compatible endpoint first
	if (hasNativeTools && hasTools) {
		voidDevLog(`[sendOllamaChatWithFallback] \u{1F680} Trying OpenAI - compatible endpoint with native tools`)

		const first = await runAttempt(params)
		if (!first.failed) {
			voidDevLog(`[sendOllamaChatWithFallback] \u{2705} OpenAI - compatible endpoint succeeded`)
			return
		}
		voidDevWarn(`[sendOllamaChatWithFallback] \u{26A0}\u{FE0F} OpenAI - compatible endpoint failed: `, first.error)

		// Transient 5xx → retry once with the same params before dropping tools.
		if (isTransientServerError(first.error)) {
			voidDevLog(`[sendOllamaChatWithFallback] \u{1F504} Detected transient server error(${first.error.status}), retrying once...`)
			await new Promise(resolve => setTimeout(resolve, 1000))
			const retry = await runAttempt(params)
			if (!retry.failed) {
				voidDevLog(`[sendOllamaChatWithFallback] \u{2705} Retry succeeded`)
				return
			}
			voidDevWarn(`[sendOllamaChatWithFallback] \u{26A0}\u{FE0F} Retry also failed: `, retry.error)
		}

		voidDevLog(`[sendOllamaChatWithFallback] \u{1F504} Retrying without native tool format`)
	}

	// Fallback: retry without tools if the native-tools attempt failed. Sending no
	// tools lets the model at least respond as a plain chat instead of erroring.
	if (hasTools) {
		const noToolParams: SendChatParams_Internal = {
			...params,
			mcpTools: [],
			acpTools: [],
			composioTools: [],
		}
		const fallback = await runAttempt(noToolParams)
		if (!fallback.failed) {
			voidDevLog(`[sendOllamaChatWithFallback] \u{2705} Fallback succeeded`)
			return
		}

		if (isTransientServerError(fallback.error)) {
			voidDevLog(`[sendOllamaChatWithFallback] \u{1F504} Detected transient server error in fallback(${fallback.error.status}), retrying once...`)
			await new Promise(resolve => setTimeout(resolve, 2000))
			const retry = await runAttempt(noToolParams)
			if (!retry.failed) {
				voidDevLog(`[sendOllamaChatWithFallback] \u{2705} Fallback retry succeeded`)
				return
			}
			voidDevWarn(`[sendOllamaChatWithFallback] \u{26A0}\u{FE0F} Fallback retry also failed: `, retry.error)
		}

		console.error(`[sendOllamaChatWithFallback] \u{274C} Both attempts failed: `, fallback.error)
		params.onError({ message: `Model produced a result A - Coder couldn't apply`, fullError: fallback.error })
	} else {
		// No tools needed, use standard OpenAI-compatible chat
		voidDevLog(`[sendOllamaChatWithFallback] 💬 No tools needed, using standard chat`)
		await runAttempt(params)
	}
}

// Implementation for Gemini using Google's native API
const sendGeminiChat = async ({
	messages,
	separateSystemMessage,
	onText,
	onFinalMessage,
	onError,
	settingsOfProvider,
	globalSettings,
	overridesOfModel,
	modelName: modelName_,
	_setAborter,
	providerName,
	modelSelectionOptions,
	chatMode,
	mcpTools,
	acpTools,
	composioTools,
	allowedTools,
	allowExternalTools,
}: SendChatParams_Internal) => {

	if (providerName !== 'gemini') throw new Error(`Sending Gemini chat, but provider was ${providerName}`)

	const thisConfig = settingsOfProvider[providerName]

	const {
		modelName,
		specialToolFormat,
		// reasoningCapabilities,
	} = getModelCapabilities(providerName, modelName_, overridesOfModel)

	// const { providerReasoningIOSettings } = getProviderCapabilities(providerName)

	// reasoning
	const reasoningInfo = getSendableReasoningInfo('Chat', providerName, modelName_, modelSelectionOptions, overridesOfModel) // user's modelName_ here

	const thinkingConfig: ThinkingConfig | undefined = !reasoningInfo?.isReasoningEnabled ? undefined
		: reasoningInfo.type === 'budget_slider_value' ?
			{ thinkingBudget: reasoningInfo.reasoningBudget }
			: undefined // Gemini only supports budget_slider, not effort_slider

	voidDevLog(`[sendLLMMessage] Gemini reasoning config:`, {
		reasoningInfo,
		thinkingConfig
	})

	// tools
	const potentialTools = geminiTools(chatMode, mcpTools, acpTools, composioTools, {
		enableMorphFastContext: (modelSelectionOptions?.morphFastContext ?? globalSettings.enableMorphFastContext) && !!globalSettings.morphApiKey,
		enableMediaGeneration: globalSettings.enableMediaGeneration,
		allowedTools,
		allowExternalTools,
	})
	const toolConfig = potentialTools && specialToolFormat === 'gemini-style' ?
		potentialTools
		: undefined

	// instance
	const genAI = new GoogleGenAI({ apiKey: thisConfig.apiKey });


	// when receive text
	let fullReasoningSoFar = ''
	let fullTextSoFar = ''

	let toolCallsAccumulator: { name: string, id: string, paramsStr: string, thoughtSignature?: string }[] = []
	let lastGeminiFinishReason: string | undefined
	// Real token usage from the final chunk's usageMetadata, used to anchor
	// context-size accounting instead of re-estimating the whole prompt.
	let finalGeminiUsage: { promptTokens: number; completionTokens: number } | undefined


	genAI.models.generateContentStream({
		model: modelName,
		config: {
			systemInstruction: separateSystemMessage,
			thinkingConfig: thinkingConfig,
			tools: toolConfig,
		},
		contents: messages as GeminiLLMChatMessage[],
	})
		.then(async (stream) => {
			_setAborter(() => { stream.return(fullTextSoFar); });

			// Process the stream
			for await (const chunk of stream) {
				// message
				const newText = chunk.text ?? ''
				fullTextSoFar += newText

			// reasoning and thought signature
			const candidates = (chunk as unknown as GeminiStreamChunkExtras).candidates
			if (candidates?.[0]?.finishReason) {
				lastGeminiFinishReason = candidates[0].finishReason
			}
			// Capture the last reported usage — Gemini streams it on most chunks.
			const geminiUsageMeta = (chunk as unknown as GeminiStreamChunkExtras).usageMetadata
			if (geminiUsageMeta && (geminiUsageMeta.promptTokenCount ?? 0) > 0) {
				finalGeminiUsage = {
					promptTokens: geminiUsageMeta.promptTokenCount ?? 0,
					completionTokens: geminiUsageMeta.candidatesTokenCount ?? 0,
				}
			}
			if (candidates?.[0]?.content?.parts?.forEach) {
				for (const part of candidates[0].content.parts) {
					if (part?.thought) {
						fullReasoningSoFar += part.text ?? ''
					}
				}
			}

				// tool call
				const functionCalls = chunk.functionCalls
				if (functionCalls && functionCalls.length > 0) {
					for (const functionCall of functionCalls) {
						const index = toolCallsAccumulator.findIndex(tc => tc.id === functionCall.id)
						if (index === -1) {
							toolCallsAccumulator.push({
								name: functionCall.name ?? '',
								id: functionCall.id ?? '',
								paramsStr: JSON.stringify(functionCall.args ?? {}),
								thoughtSignature: (functionCall as unknown as GeminiFunctionCallExtras).thought_signature
							})
						} else {
							// Update existing call (though Gemini usually sends full calls)
							toolCallsAccumulator[index].paramsStr = JSON.stringify(functionCall.args ?? {})
						}
					}
				}

				// call onText
				onText({
					fullText: fullTextSoFar,
					fullReasoning: fullReasoningSoFar,
					toolCalls: toolCallsAccumulator.length === 0 ? undefined : toolCallsAccumulator.map(tc => ({
						name: tc.name,
						rawParams: parsePartialJSON(tc.paramsStr),
						isDone: false,
						doneParams: [],
						id: tc.id || generateUuid()
					})),
				})
			}

			// on final
			if (!fullTextSoFar && !fullReasoningSoFar && toolCallsAccumulator.length === 0) {
				onError({ message: 'A-Coder IDE: Response from model was empty.', fullError: null })
			} else {
				const finalToolCalls = toolCallsAccumulator.map(tc => rawToolCallObjOfParamsStr(tc.name, tc.paramsStr, tc.id || generateUuid(), tc.thoughtSignature)).filter(tc => !!tc) as RawToolCallObj[]
				const toolCallObj = finalToolCalls.length > 0 ? { toolCalls: finalToolCalls } : {}
				const anthropicReasoning: AnthropicReasoning[] | null = fullReasoningSoFar ? [{ type: 'thinking', thinking: fullReasoningSoFar, signature: toolCallsAccumulator[0]?.thoughtSignature || '' }] : null
				onFinalMessage({ fullText: fullTextSoFar, fullReasoning: fullReasoningSoFar, anthropicReasoning, ...toolCallObj, usage: finalGeminiUsage, stopReason: lastGeminiFinishReason });
			}
		})
		.catch(error => {
			const message = error?.message
			if (typeof message === 'string') {

				if (error.message?.includes('API key')) {
					onError({ message: invalidApiKeyMessage(providerName), fullError: error });
				}
				else if (error?.message?.includes('429')) {
					onError({ message: 'Rate limit reached. ' + error, fullError: error });
				}
				else
					onError({ message: error + '', fullError: error });
			}
			else {
				onError({ message: error + '', fullError: error });
			}
		})
};



type CallFnOfProvider = {
	[providerName in ProviderName]: {
		sendChat: (params: SendChatParams_Internal) => Promise<void>;
		sendFIM: ((params: SendFIMParams_Internal) => void) | null;
		list: ((params: ListParams_Internal<any>) => void) | null;
	}
}

export const sendLLMMessageToProviderImplementation = {
	anthropic: {
		sendChat: sendAnthropicChat,
		sendFIM: null,
		list: null,
	},
	openAI: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: null,
	},
	xAI: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: null,
	},
	gemini: {
		sendChat: (params) => sendGeminiChat(params),
		sendFIM: null,
		list: null,
	},
	mistral: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: (params) => sendMistralFIM(params),
		list: null,
	},
	ollama: {
		sendChat: (params) => _sendOllamaChatWithFallback(params),
		sendFIM: sendOllamaFIM,
		list: ollamaList,
	},
	ollamaCloud: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: (params) => _ollamaCloudList(params),
	},
	openAICompatible: {
		sendChat: (params) => _sendOpenAICompatibleChat(params), // using openai's SDK is not ideal (your implementation might not do tools, reasoning, FIM etc correctly), talk to us for a custom integration
		sendFIM: (params) => _sendOpenAICompatibleFIM(params),
		list: null,
	},
	openRouter: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: (params) => _sendOpenAICompatibleFIM(params),
		list: null,
	},
	vLLM: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: (params) => _sendOpenAICompatibleFIM(params),
		list: (params) => _openaiCompatibleList(params),
	},
	deepseek: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: null,
	},
	groq: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: null,
	},

	lmStudio: {
		// lmStudio has no suffix parameter in /completions, so sendFIM might not work
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: (params) => _sendOpenAICompatibleFIM(params),
		list: (params) => _openaiCompatibleList(params),
	},
	liteLLM: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: (params) => _sendOpenAICompatibleFIM(params),
		list: null,
	},
	googleVertex: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: null,
	},
	microsoftAzure: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: null,
	},
	awsBedrock: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: null,
	},
	aCoder: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: (params) => _openaiCompatibleList(params),
	},
	openAdapter: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: null,
		list: (params) => _openaiCompatibleList(params),
	},
	llamaCpp: {
		sendChat: (params) => _sendOpenAICompatibleChat(params),
		sendFIM: (params) => _sendOpenAICompatibleFIM(params),
		list: (params) => _openaiCompatibleList(params),
	},
} satisfies CallFnOfProvider

/*
FIM info (this may be useful in the future with vLLM, but in most cases the only way to use FIM is if the provider explicitly supports it):

qwen2.5-coder https://ollama.com/library/qwen2.5-coder/blobs/e94a8ecb9327
<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>

codestral https://ollama.com/library/codestral/blobs/51707752a87c
[SUFFIX]{{ .Suffix }}[PREFIX] {{ .Prompt }}

deepseek-coder-v2 https://ollama.com/library/deepseek-coder-v2/blobs/22091531faf0
<｜fim▁begin｜>{{ .Prompt }}<｜fim▁hole｜>{{ .Suffix }}<｜fim▁end｜>

starcoder2 https://ollama.com/library/starcoder2/blobs/3b190e68fefe
<file_sep>
<fim_prefix>
{{ .Prompt }}<fim_suffix>{{ .Suffix }}<fim_middle>
<|end_of_text|>

codegemma https://ollama.com/library/codegemma:2b/blobs/48d9a8140749
<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>

*/
