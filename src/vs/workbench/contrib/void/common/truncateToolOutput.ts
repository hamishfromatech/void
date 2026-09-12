/*--------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

/**
 * Shared truncation utilities for tool outputs (ported from a-coder-cli's
 * truncate.ts doctrine).
 *
 * Truncation is based on two independent limits - whichever is hit first wins:
 * - Line limit (default: 2000 lines)
 * - Byte limit (default: 50KB)
 *
 * Never returns partial lines: output is always cut on a line boundary so the
 * LLM sees intact rows and line-number references stay correct.
 */

export const DEFAULT_MAX_LINES = 2000;
export const DEFAULT_MAX_BYTES = 50 * 1024; // 50KB

export interface TruncationOptions {
	/** Maximum number of lines (default: 2000) */
	maxLines?: number;
	/** Maximum number of UTF-8 bytes (default: 50KB) */
	maxBytes?: number;
}

export interface TruncationResult {
	/** The truncated content (without any truncation notice) */
	content: string;
	/** Whether truncation occurred */
	truncated: boolean;
	/** Which limit was hit: "lines", "bytes", or null if not truncated */
	truncatedBy: 'lines' | 'bytes' | null;
	/** Total number of lines in the original content */
	totalLines: number;
	/** Total number of UTF-8 bytes in the original content */
	totalBytes: number;
	/** Number of complete lines in the truncated output */
	outputLines: number;
	/** Number of UTF-8 bytes in the truncated output */
	outputBytes: number;
	/**
	 * True when the line at the cut boundary exceeded the byte budget by itself
	 * (head truncation edge case: even a single further line didn't fit).
	 */
	lastLinePartial: boolean;
}

/** UTF-8 byte length without allocating (surrogate-pair aware). */
function utf8ByteLength(s: string): number {
	let bytes = 0;
	for (let i = 0; i < s.length; i++) {
		const code = s.charCodeAt(i);
		if (code < 0x80) bytes += 1;
		else if (code < 0x800) bytes += 2;
		else if (code >= 0xD800 && code <= 0xDBFF) { bytes += 4; i++; } // surrogate pair counts once
		else bytes += 3;
	}
	return bytes;
}

/**
 * Truncate `content` to at most `maxLines` lines and `maxBytes` UTF-8 bytes,
 * whichever limit is hit first, always on a line boundary.
 */
export function truncateToolOutput(content: string, options: TruncationOptions = {}): TruncationResult {
	const maxLines = options.maxLines ?? DEFAULT_MAX_LINES;
	const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;

	const lines = content.split('\n');
	const totalLines = lines.length;

	// Fast path: nothing to do when both limits are comfortably satisfied.
	const totalBytes = utf8ByteLength(content);
	if (lines.length <= maxLines && totalBytes <= maxBytes) {
		return {
			content,
			truncated: false,
			truncatedBy: null,
			totalLines,
			totalBytes,
			outputLines: lines.length,
			outputBytes: totalBytes,
			lastLinePartial: false,
		};
	}

	let cutIndex = -1; // exclusive end of kept lines
	let truncatedBy: 'lines' | 'bytes' | null = null;
	let bytesSoFar = 0;
	let lastLinePartial = false;

	for (let i = 0; i < lines.length; i++) {
		if (i >= maxLines) {
			truncatedBy = 'lines';
			cutIndex = i;
			break;
		}
		const lineBytes = utf8ByteLength(lines[i]) + (i < lines.length - 1 ? 1 : 0); // +1 newline except last
		if (bytesSoFar + lineBytes > maxBytes) {
			truncatedBy = 'bytes';
			cutIndex = i;
			// The line alone is bigger than the whole budget (or the remainder):
			// mark it as the partially-shown boundary.
			lastLinePartial = lineBytes - 1 > maxBytes;
			break;
		}
		bytesSoFar += lineBytes;
	}

	if (cutIndex === -1 || cutIndex === 0) {
		// First line alone exceeds the byte budget — keep a bounded head slice of
		// it so the LLM still gets something (this is the one intentional
		// partial-line edge case; the notice below makes it explicit).
		// Slice by UTF-8 bytes, not UTF-16 units: narrow to fit, then verify.
		let headUnits = maxBytes; // 1 unit ≥ 1 byte, so this is an upper bound
		let head = content.slice(0, headUnits);
		let headBytes = utf8ByteLength(head);
		while (headBytes > maxBytes && headUnits > 1) {
			headUnits = Math.max(1, Math.floor(headUnits * maxBytes / headBytes) - 1);
			head = content.slice(0, headUnits);
			headBytes = utf8ByteLength(head);
		}
		return {
			content: head,
			truncated: true,
			truncatedBy: 'bytes',
			totalLines,
			totalBytes,
			outputLines: 1,
			outputBytes: headBytes,
			lastLinePartial: true,
		};
	}

	const keptLines = lines.slice(0, cutIndex);
	const kept = keptLines.join('\n');
	return {
		content: kept,
		truncated: true,
		truncatedBy,
		totalLines,
		totalBytes,
		outputLines: keptLines.length,
		outputBytes: utf8ByteLength(kept),
		lastLinePartial,
	};
}

/**
 * Truncate and append an explicit truncation notice telling the model what
 * happened and how to get more — mirroring a-coder-cli's tool output format.
 */
export function truncateToolOutputWithNotice(content: string, options: TruncationOptions = {}): string {
	const result = truncateToolOutput(content, options);
	if (!result.truncated) return result.content;

	const limitDesc = result.truncatedBy === 'lines'
		? `line limit of ${options.maxLines ?? DEFAULT_MAX_LINES} lines`
		: `size limit of ${Math.round((options.maxBytes ?? DEFAULT_MAX_BYTES) / 1024)}KB`;

	const more = result.outputLines < result.totalLines
		? ` (showing ${result.outputLines} of ${result.totalLines} lines)`
		: '';

	return `${result.content}\n\n[Output truncated by ${limitDesc}${more}. ${result.lastLinePartial ? 'A single line exceeded the limit, so it is partially shown. ' : ''}Re-run with narrower parameters (line ranges, filters) to see the rest.]`;
}