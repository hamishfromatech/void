/*---------------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------------*/

import assert from 'assert';
import {
	truncateToolOutput,
	truncateToolOutputWithNotice,
	DEFAULT_MAX_LINES,
	DEFAULT_MAX_BYTES,
} from '../../common/truncateToolOutput.js';

suite('truncateToolOutput', () => {

	test('returns content unchanged when under both limits', () => {
		const content = 'line1\nline2\nline3';
		const result = truncateToolOutput(content);
		assert.strictEqual(result.truncated, false);
		assert.strictEqual(result.truncatedBy, null);
		assert.strictEqual(result.content, content);
		assert.strictEqual(result.totalLines, 3);
		assert.strictEqual(result.totalBytes, content.length); // ascii
	});

	test('truncates on the line limit, never mid-line', () => {
		const lines = Array.from({ length: 100 }, (_, i) => `line ${i}`);
		const content = lines.join('\n');
		const result = truncateToolOutput(content, { maxLines: 10 });
		assert.strictEqual(result.truncated, true);
		assert.strictEqual(result.truncatedBy, 'lines');
		assert.strictEqual(result.content, lines.slice(0, 10).join('\n'));
		assert.strictEqual(result.outputLines, 10);
		assert.strictEqual(result.totalLines, 100);
		assert.strictEqual(result.lastLinePartial, false);
	});

	test('truncates on the byte limit at a line boundary', () => {
		const lines = ['a'.repeat(100), 'b'.repeat(100), 'c'.repeat(100), 'd'.repeat(100)];
		const content = lines.join('\n');
		// Budget fits exactly 3 lines + 3 newlines
		const result = truncateToolOutput(content, { maxBytes: 100 * 3 + 3 });
		assert.strictEqual(result.truncated, true);
		assert.strictEqual(result.truncatedBy, 'bytes');
		assert.strictEqual(result.content, lines.slice(0, 3).join('\n'));
		assert.strictEqual(result.outputLines, 3);
		assert.strictEqual(result.lastLinePartial, false);
	});

	test('whichever limit hits first wins', () => {
		const longLines = Array.from({ length: 5000 }, (_, i) => `x${i}`);
		const result = truncateToolOutput(longLines.join('\n'), { maxLines: 10, maxBytes: DEFAULT_MAX_BYTES });
		assert.strictEqual(result.truncatedBy, 'lines');
		const bigLines = Array.from({ length: 5 }, () => 'z'.repeat(40_000));
		const result2 = truncateToolOutput(bigLines.join('\n'), { maxLines: DEFAULT_MAX_LINES, maxBytes: DEFAULT_MAX_BYTES });
		assert.strictEqual(result2.truncatedBy, 'bytes');
	});

	test('utf-8 multibyte content counts bytes not chars', () => {
		// 'é' is 2 bytes utf-8, 1 utf-16 unit
		const content = 'é'.repeat(60_000); // 120,000 utf-8 bytes
		const result = truncateToolOutput(content, { maxBytes: 1000 });
		assert.strictEqual(result.truncated, true);
		assert.strictEqual(result.truncatedBy, 'bytes');
		assert.strictEqual(result.totalBytes, 120_000);
		assert.ok(result.outputBytes <= 1000);
	});

	test('single line exceeding byte budget keeps a bounded head', () => {
		const content = 'y'.repeat(10_000);
		const result = truncateToolOutput(content, { maxBytes: 100 });
		assert.strictEqual(result.truncated, true);
		assert.strictEqual(result.lastLinePartial, true);
		assert.strictEqual(result.outputLines, 1);
		assert.ok(result.content.length <= 100);
	});

	test('notice wrapper appends guidance only when truncated', () => {
		const short = 'all good';
		assert.strictEqual(truncateToolOutputWithNotice(short), short);

		const lines = Array.from({ length: DEFAULT_MAX_LINES + 10 }, (_, i) => `row ${i}`);
		const noticed = truncateToolOutputWithNotice(lines.join('\n'));
		assert.ok(noticed.includes('[Output truncated by line limit'));
		assert.ok(noticed.includes(`showing ${DEFAULT_MAX_LINES} of ${lines.length} lines`));
		assert.ok(noticed.startsWith('row 0'));
	});

	test('trailing newline does not produce a phantom truncation', () => {
		const content = 'one\ntwo\n';
		const result = truncateToolOutput(content, { maxLines: 10, maxBytes: 1000 });
		assert.strictEqual(result.truncated, false);
		assert.strictEqual(result.content, content);
	});
});