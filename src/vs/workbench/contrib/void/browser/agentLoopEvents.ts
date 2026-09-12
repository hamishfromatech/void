/*--------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

/**
 * Typed events emitted by the chat agent loop (ported from a-coder-cli's
 * AgentEventSink architecture). The loop currently lives inline in
 * ChatThreadService; these events give it a typed, subscribeable surface so
 * instrumentation, extensions, and future extraction of the loop into a pure
 * module can consume loop transitions without reaching into stream state.
 *
 * Emission is strictly observational: firing must never throw into the loop.
 */
export type AgentLoopEvent =
	| { type: 'agent_run_started'; threadId: string; chatMode: string }
	| { type: 'agent_turn_started'; threadId: string; turnNumber: number }
	| { type: 'llm_request_started'; threadId: string; messageCount: number }
	| { type: 'llm_response_received'; threadId: string; toolCallCount: number; stopReason?: string }
	| { type: 'llm_errored'; threadId: string; errorMessage: string }
	| { type: 'llm_aborted'; threadId: string }
	| { type: 'tool_execution_started'; threadId: string; toolName: string }
	| { type: 'tool_execution_finished'; threadId: string; toolName: string; isError: boolean }
	| { type: 'agent_run_max_iterations'; threadId: string; maxIterations: number }
	| { type: 'agent_run_finished'; threadId: string; outcome: 'done' | 'error' | 'aborted' }

export type AgentEventSink = (event: AgentLoopEvent) => void;