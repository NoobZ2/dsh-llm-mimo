/**
 * dsh-llm-mimo — Xiaomi MiMo provider route for DeepSeek Harness.
 *
 * Registers the `xiaomi-mimo` provider on `ctx.llm`, backed by the official
 * Xiaomi MiMo OpenAI-compatible endpoint (https://api.xiaomimimo.com/v1).
 *
 * Wire capabilities (verified against the live API):
 *   - streaming chat completions with `stream_options.include_usage`
 *   - `reasoning_content` deltas (MiMo deep-thinking mode) → harness reasoning blocks
 *   - standard OpenAI `tool_calls` deltas (finish_reason `tool_calls`)
 *   - full-modal image input on `mimo-v2.5` (base64 data URLs)
 *
 * Image serialization rules (MiMo rejects image parts inside `role: tool`
 * messages): images in user content serialize in place as `image_url` parts;
 * images inside tool-result content (e.g. the `read_image` tool) are hoisted
 * into the next user wire message so they still reach the model.
 *
 * Connection facts (base URL, catalog, key) are resolved per request through
 * the `llm-mimo` settings section and the credentials seam, so edits apply to
 * the very next request without a restart.
 */
import z from "@deepseek-ai/schemastery";
import {
	CONTEXT_WINDOW_EXCEEDED_CODE,
	CallId,
	EMPTY_RESPONSE_CODE,
	LlmAdapter,
	LlmError,
	ProviderRequestId,
	ReasoningEffortId,
	QUOTA_EXCEEDED_CODE,
	RetryPolicySchema,
	assertUsableApiKey,
	attributionHeaders,
	isContextWindowExceededError,
	isQuotaExceededError,
	resolveRetryPolicy,
} from "@deepseek-ai/dsh-llm";
import { credentialRef } from "@deepseek-ai/dsh-credentials";
import { launchEnvironmentOf } from "@deepseek-ai/dsh-launch-environment";
import { deepEqualJson, installSettingsSection, settingsNamespace } from "@deepseek-ai/dsh-settings";
import { MAX_TIMER_DELAY_MS, idleWatchdog, timeoutOf } from "@deepseek-ai/dsh-timeout";
import { getOrCreateAnonymousUserId } from "@deepseek-ai/dsh-anonymous-user-id";
import { EventSourceParserStream } from "eventsource-parser/stream";

// ── constants ────────────────────────────────────────────────────────────────
const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 3e5;
const DEFAULT_CONTEXT_WINDOW = 1_048_576; // 1M, per MiMo model card
const DEFAULT_MAX_TOKENS = 131_072; // 128K max output, per MiMo model card
const STREAM_IDLE_TIMEOUT_CODE = "LLM_STREAM_IDLE_TIMEOUT";
const OFF_REASONING_EFFORT = ReasoningEffortId("off");
const HIGH_REASONING_EFFORT = ReasoningEffortId("high");
const MAX_REASONING_EFFORT = ReasoningEffortId("max");
/**
 * The harness validates the request's reasoningEffort against this list, and
 * the parent session header usually carries `max`. MiMo has no wire
 * reasoning-effort knob (its deep thinking is automatic), so these ids are
 * declared for compatibility and never serialized.
 */
const REASONING_EFFORTS = [
	{ id: OFF_REASONING_EFFORT, name: "Off" },
	{ id: HIGH_REASONING_EFFORT, name: "High" },
	{ id: MAX_REASONING_EFFORT, name: "Max" },
];

const name = "llm-mimo";
const inject = ["llm"];
const NS = settingsNamespace("llm-mimo");
const DEFAULT_API_KEY_ENV = "MIMO_API_KEY";
const PUBLIC_BASE_URL = "https://api.xiaomimimo.com/v1";
const BASE_URL_ENV = "MIMO_BASE_URL";
/** The single provider route this plugin owns. */
const PROVIDER = "xiaomi-mimo";

const DEFAULT_MODELS = [
	{
		id: "mimo-v2.5",
		name: "MiMo-V2.5",
		description: "Xiaomi MiMo-V2.5, full-modal understanding (image + text), deep thinking, function calling.",
		contextWindow: DEFAULT_CONTEXT_WINDOW,
		maxTokens: DEFAULT_MAX_TOKENS,
		inputModalities: ["text", "image"],
	},
	{
		id: "mimo-v2.5-pro",
		name: "MiMo-V2.5-Pro",
		description: "Xiaomi MiMo-V2.5-Pro, text generation, deep thinking, function calling.",
		contextWindow: DEFAULT_CONTEXT_WINDOW,
		maxTokens: DEFAULT_MAX_TOKENS,
		inputModalities: ["text"],
	},
];

// ── config schema ────────────────────────────────────────────────────────────
const catalogModel = z.object({
	id: z.string().required(),
	name: z.string(),
	description: z.string(),
	contextWindow: z.number().step(1).min(1),
	maxTokens: z.number().step(1).min(1),
	inputModalities: z.array(z.string()),
});

const Config = z.object({
	apiKeyEnv: z.string().role("credential-ref").default(DEFAULT_API_KEY_ENV),
	baseURL: z.string(),
	maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(DEFAULT_MAX_TOKENS),
	defaultContextWindow: z.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
	models: z.array(catalogModel).default(DEFAULT_MODELS),
	streamIdleTimeoutMs: z.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
	retryPolicy: RetryPolicySchema,
});

/** Resolve, validate, and detach the advisory model catalog. */
function resolveModels(models) {
	const seen = new Set();
	return (models ?? DEFAULT_MODELS).map((model) => {
		if (model.id.length === 0) throw new Error("llm-mimo: catalog model ids must be non-empty");
		if (model.name !== void 0 && model.name.length === 0) throw new Error(`llm-mimo: catalog model "${model.id}" has an empty name`);
		if (model.contextWindow !== void 0 && (!Number.isInteger(model.contextWindow) || model.contextWindow <= 0)) {
			throw new Error(`llm-mimo: catalog model "${model.id}" contextWindow must be a positive integer`);
		}
		if (model.maxTokens !== void 0 && (!Number.isInteger(model.maxTokens) || model.maxTokens <= 0)) {
			throw new Error(`llm-mimo: catalog model "${model.id}" maxTokens must be a positive integer`);
		}
		const modalities = model.inputModalities ?? ["text"];
		if (!Array.isArray(modalities) || modalities.length === 0 || !modalities.includes("text")) {
			throw new Error(`llm-mimo: catalog model "${model.id}" inputModalities must be a non-empty array containing "text"`);
		}
		if (seen.has(model.id)) throw new Error(`llm-mimo: duplicate catalog model "${model.id}"`);
		seen.add(model.id);
		return {
			id: model.id,
			...model.name === void 0 ? {} : { name: model.name },
			...model.description === void 0 ? {} : { description: model.description },
			...model.contextWindow === void 0 ? {} : { contextWindow: model.contextWindow },
			...model.maxTokens === void 0 ? {} : { maxTokens: model.maxTokens },
			inputModalities: [...modalities],
		};
	});
}

/** Resolve raw config to validated connection facts (fail loud at load, keep-last-good on settings edits). */
function resolveAdapterOptions(config, environment) {
	if (config.defaultContextWindow !== void 0 && (!Number.isInteger(config.defaultContextWindow) || config.defaultContextWindow <= 0)) {
		throw new Error("llm-mimo: defaultContextWindow must be a positive integer");
	}
	if (config.maxTokens !== void 0 && (!Number.isSafeInteger(config.maxTokens) || config.maxTokens <= 0)) {
		throw new Error("llm-mimo: maxTokens must be a positive safe integer");
	}
	const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS;
	if (!Number.isFinite(streamIdleTimeoutMs) || streamIdleTimeoutMs <= 0 || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
		throw new Error(`llm-mimo: streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`);
	}
	return {
		apiKeyEnv: credentialRef(config.apiKeyEnv ?? DEFAULT_API_KEY_ENV),
		baseURL: config.baseURL ?? environment?.get(BASE_URL_ENV)?.value ?? PUBLIC_BASE_URL,
		maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
		defaultContextWindow: config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW,
		models: resolveModels(config.models),
		streamIdleTimeoutMs,
		retryPolicy: resolveRetryPolicy(config.retryPolicy, "llm-mimo: retryPolicy"),
	};
}

// ── serialization ────────────────────────────────────────────────────────────
/** Join the text blocks of a message (used for system / tool-result content). */
function flattenText(blocks) {
	return blocks.filter((block) => block.type === "text").map((block) => block.text).join("");
}

/** Recursive image detection: image blocks may nest inside tool-result blocks. */
function blocksHaveImage(blocks) {
	return blocks.some((block) => block.type === "image" || block.type === "tool-result" && blocksHaveImage(block.content));
}

/** Serialize one assistant message (text + reasoning + tool calls). */
function serializeAssistant(message) {
	const text = flattenText(message.content);
	const reasoning = message.content.filter((block) => block.type === "reasoning").map((block) => block.text).join("");
	const toolCalls = message.content.filter((block) => block.type === "tool-call").map((block) => ({
		id: block.id,
		type: "function",
		function: { name: block.name, arguments: block.arguments },
	}));
	return {
		role: "assistant",
		content: text,
		...toolCalls.length > 0 && reasoning.length > 0 ? { reasoning_content: reasoning } : {},
		...toolCalls.length > 0 ? { tool_calls: toolCalls } : {},
	};
}

/** Convert one harness image block into an OpenAI `image_url` part via the durable attachment service. */
async function imagePart(block, attachments) {
	const ref = block.attachment ?? (block.attachmentId !== void 0 ? block : void 0);
	if (ref === void 0) throw new LlmError("llm-mimo: image block carries no attachment reference", "UNSUPPORTED_CONTENT");
	const stored = await attachments.readImage(ref);
	const mediaType = stored.ref?.mediaType ?? ref.mediaType ?? "image/png";
	return {
		type: "image_url",
		image_url: { url: `data:${mediaType};base64,${Buffer.from(stored.data).toString("base64")}` },
	};
}

/** Collect wire image parts for one block list: image blocks (and, recursively, images inside tool results). */
async function collectParts(blocks, attachments) {
	const parts = [];
	for (const block of blocks) {
		if (block.type === "image") {
			parts.push(await imagePart(block, attachments));
		} else if (block.type === "tool-result") {
			parts.push(...await collectParts(block.content, attachments));
		}
	}
	return parts;
}

/**
 * Sanitize a JSON Schema for MiMo's validator. MiMo rejects tuple-style
 * `items` arrays (observed on MCP-generated schemas such as
 * `mcp__obsidian__obsidian_read_pdf`): a single `items` object is required.
 * Tuples are preserved as `anyOf` alternatives. Other constructs
 * (`$schema`, `pattern`, `exclusiveMinimum`, ...) are accepted as-is.
 */
function sanitizeSchema(schema) {
	if (schema === null || typeof schema !== "object" || Array.isArray(schema)) return schema;
	const out = {};
	for (const [key, value] of Object.entries(schema)) {
		if (key === "items" && Array.isArray(value)) {
			if (value.length === 0) {
				out.items = {};
			} else if (value.length === 1) {
				out.items = sanitizeSchema(value[0]);
			} else {
				out.items = { anyOf: value.map((entry) => sanitizeSchema(entry)) };
			}
			continue;
		}
		out[key] = typeof value === "object" && value !== null ? sanitizeSchema(value) : value;
	}
	return out;
}

/**
 * Serialize the conversation into OpenAI chat-completions wire messages.
 * User images serialize in place; images found inside tool-result content are
 * hoisted into the next user wire message (MiMo rejects image parts in
 * `role: tool` messages). A trailing hoisted image gets a final user message.
 * @param messages - the harness conversation, in order.
 * @param attachments - the durable attachment service (required when images are present).
 */
async function serializeMessages(messages, attachments) {
	const wire = [];
	let pendingImages = [];
	for (const message of messages) {
		if (message.role === "system") {
			wire.push({ role: "system", content: flattenText(message.content) });
			continue;
		}
		if (message.role === "assistant") {
			if (message.content.some((block) => block.type === "image")) {
				throw new LlmError("llm-mimo: cannot represent structured assistant image output", "UNSUPPORTED_CONTENT");
			}
			wire.push(serializeAssistant(message));
			continue;
		}
		// user-role message: text + optional user images, then its tool results
		const toolResults = message.content.filter((block) => block.type === "tool-result");
		const toolImages = await collectParts(toolResults, attachments);
		const direct = await collectParts(message.content.filter((block) => block.type !== "tool-result"), attachments);
		const text = flattenText(message.content);
		const hoisted = pendingImages;
		pendingImages = toolImages;
		const parts = [...hoisted, ...direct];
		if (text.length > 0 || toolResults.length === 0) {
			const content = parts.length === 0 ? text : [...parts, ...(text.length > 0 ? [{ type: "text", text }] : [])];
			if (typeof content === "string" || content.length > 0) wire.push({ role: "user", content });
		}
		for (const result of toolResults) {
			wire.push({ role: "tool", tool_call_id: result.toolCallId, content: flattenText(result.content) || "(no output)" });
		}
	}
	if (pendingImages.length > 0) {
		wire.push({ role: "user", content: [...pendingImages, { type: "text", text: "[The image(s) above were attached by the previous tool call.]" }] });
	}
	return wire;
}

/** Build the full wire request (always streaming, usage reporting on). */
async function serializeRequest(options, defaults, attachments, connection) {
	const messages = [];
	if (options.system !== void 0) messages.push({ role: "system", content: options.system });
	messages.push(...await serializeMessages(options.messages, attachments));
	const tools = options.tools?.map((tool) => ({
		type: "function",
		function: { name: tool.name, description: tool.description, parameters: sanitizeSchema(tool.parameters) },
	}));
	// Clamp the harness-requested max_tokens to the model's catalog cap: the
	// parent session header usually carries 256k, while MiMo's max output is
	// 128k and exceeding it is a 400.
	const model = connection?.models.find((entry) => entry.id === options.model);
	const cap = model?.maxTokens ?? connection?.maxTokens ?? DEFAULT_MAX_TOKENS;
	return {
		model: options.model,
		messages,
		stream: true,
		stream_options: { include_usage: true },
		...tools !== void 0 && tools.length > 0 ? { tools } : {},
		...options.temperature !== void 0 ? { temperature: options.temperature } : {},
		...options.maxTokens === void 0 ? {} : { max_tokens: Math.min(options.maxTokens, cap) },
		...options.stop !== void 0 ? { stop: options.stop } : {},
	};
}

// ── SSE parsing ──────────────────────────────────────────────────────────────
/** Parse an SSE byte stream into data payloads; `[DONE]` is the final value. */
async function* parseSse(stream, onComment) {
	const events = stream.pipeThrough(new TextDecoderStream()).pipeThrough(new EventSourceParserStream({ onComment }));
	for await (const { data } of events) {
		yield data;
		if (data === "[DONE]") return;
	}
	throw new LlmError("SSE stream ended without [DONE]", "STREAM_CLOSED");
}

// ── translate ────────────────────────────────────────────────────────────────
/** Map the wire finish_reason vocabulary to the harness FinishReason. */
function mapFinishReason(reason) {
	switch (reason) {
		case "stop": return { kind: "stop" };
		case "tool_calls": return { kind: "tool-calls" };
		case "length": return { kind: "max-tokens" };
		default: return { kind: "error", failure: { message: `model stopped: ${reason}`, code: reason.toUpperCase() } };
	}
}

/** Map wire usage to disjoint harness counts (cache reads and reasoning tokens split out). */
function mapUsage(usage) {
	const cacheRead = usage.prompt_tokens_details?.cached_tokens ?? usage.prompt_cache_hit_tokens;
	const reasoning = usage.completion_tokens_details?.reasoning_tokens;
	return {
		inputTokens: usage.prompt_tokens - (cacheRead ?? 0),
		outputTokens: usage.completion_tokens,
		...cacheRead !== void 0 ? { cacheReadTokens: cacheRead } : {},
		...reasoning !== void 0 ? { reasoningTokens: reasoning } : {},
	};
}

/** Assemble the final ContentBlock for one open block. */
function closeBlock(block) {
	switch (block.kind) {
		case "text": return { type: "text", text: block.text };
		case "reasoning": return { type: "reasoning", text: block.text };
		case "tool-call": return { type: "tool-call", id: CallId(block.callId ?? ""), name: block.name ?? "", arguments: block.text };
	}
}

/** Translate MiMo SSE payloads into harness StreamChunks. */
async function* translate(payloads) {
	let nextIndex = 0;
	let textBlock;
	let reasoningBlock;
	const toolBlocks = new Map();
	const order = [];
	let pendingFinish;
	let pendingUsage;
	function open(kind) {
		const block = { index: nextIndex++, kind, text: "" };
		order.push(block);
		return block;
	}
	for await (const payload of payloads) {
		if (payload === "[DONE]") {
			for (const block of order) yield { type: "block-end", index: block.index, block: closeBlock(block) };
			if (pendingUsage) yield { type: "usage", usage: pendingUsage };
			const reason = pendingFinish ?? { kind: "stop" };
			yield {
				type: "finish",
				reason: reason.kind === "stop" && order.length === 0 ? {
					kind: "error",
					failure: { message: "model returned a completed response with no content", code: EMPTY_RESPONSE_CODE },
				} : reason,
			};
			return;
		}
		let chunk;
		try {
			chunk = JSON.parse(payload);
		} catch {
			throw new LlmError(`malformed SSE payload: ${payload.slice(0, 120)}`, "MALFORMED_RESPONSE");
		}
		for (const choice of chunk.choices ?? []) {
			const delta = choice.delta ?? {};
			const reasoning = delta.reasoning_content;
			if (typeof reasoning === "string" && reasoning.length > 0) {
				if (!reasoningBlock) {
					reasoningBlock = open("reasoning");
					yield { type: "block-start", index: reasoningBlock.index, blockType: "reasoning" };
				}
				reasoningBlock.text += reasoning;
				yield { type: "reasoning-delta", index: reasoningBlock.index, text: reasoning };
			}
			const content = delta.content;
			if (typeof content === "string" && content.length > 0) {
				if (!textBlock) {
					textBlock = open("text");
					yield { type: "block-start", index: textBlock.index, blockType: "text" };
				}
				textBlock.text += content;
				yield { type: "text-delta", index: textBlock.index, text: content };
			}
			for (const call of delta.tool_calls ?? []) {
				let block = toolBlocks.get(call.index);
				if (!block) {
					block = open("tool-call");
					toolBlocks.set(call.index, block);
					yield { type: "block-start", index: block.index, blockType: "tool-call" };
				}
				// MiMo streams trailing deltas with `id: null` / `name: null`;
				// only non-empty strings may overwrite the recorded values.
				if (typeof call.id === "string" && call.id.length > 0) block.callId = call.id;
				if (typeof call.function?.name === "string" && call.function.name.length > 0) block.name = call.function.name;
				const fragment = typeof call.function?.arguments === "string" ? call.function.arguments : "";
				block.text += fragment;
				yield {
					type: "tool-call-delta",
					index: block.index,
					id: CallId(block.callId ?? ""),
					...block.name !== void 0 ? { name: block.name } : {},
					argumentsDelta: fragment,
				};
			}
			if (typeof choice.finish_reason === "string") pendingFinish = mapFinishReason(choice.finish_reason);
		}
		if (chunk.usage) pendingUsage = mapUsage(chunk.usage);
	}
	throw new LlmError("SSE payload stream ended without [DONE]", "STREAM_CLOSED");
}

// ── adapter ──────────────────────────────────────────────────────────────────
function modelInfo(provider, model) {
	return {
		provider,
		id: model.id,
		name: model.name ?? model.id,
		...model.description === void 0 ? {} : { description: model.description },
		inputModalities: model.inputModalities ?? ["text"],
	};
}

function providerRetryAfterMs(value) {
	if (value === null) return void 0;
	if (/^\d+$/.test(value)) {
		const delay = Number(value) * 1e3;
		return Number.isFinite(delay) && delay > 0 ? delay : void 0;
	}
	const delay = Date.parse(value) - Date.now();
	return Number.isFinite(delay) && delay > 0 ? delay : void 0;
}

function requestId(headers) {
	const value = headers.get("x-request-id") ?? headers.get("x-mimo-request-id");
	return value === null || value.length === 0 ? void 0 : ProviderRequestId(value);
}

/** Map an HTTP status to a stable LlmError code. */
function httpErrorCode(status, error) {
	if (status === 401 || status === 403) return "AUTH";
	const detail = [error?.code, error?.type, error?.message].filter(Boolean).join(" ");
	if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE;
	if (status === 429) return "RATE_LIMIT";
	if (status === 400) {
		if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE;
		return "INVALID_REQUEST";
	}
	if (status >= 500) return "SERVER";
	return `HTTP_${status}`;
}

/** The MiMo adapter: fetch + SSE against the Xiaomi MiMo OpenAI-compatible endpoint. */
var MiMoAdapter = class extends LlmAdapter {
	config;
	constructor(config) {
		super();
		this.config = config;
	}
	providerInfo(provider) {
		return { id: provider, name: "Xiaomi MiMo" };
	}
	providerRetryPolicy(_provider) {
		return this.config.options().retryPolicy;
	}
	listModels(provider) {
		return Promise.resolve(this.config.options().models.map((model) => modelInfo(provider, model)));
	}
	resolveModel(provider, model, _signal) {
		const connection = this.config.options();
		const configured = connection.models.find((entry) => entry.id === model);
		const contextWindow = configured?.contextWindow ?? connection.defaultContextWindow;
		return Promise.resolve({
			...configured === void 0 ? { provider, id: model, name: model, inputModalities: ["text"] } : modelInfo(provider, configured),
			context: { contextWindow },
			defaultMaxTokens: configured?.maxTokens ?? connection.maxTokens,
			reasoning: { efforts: REASONING_EFFORTS, defaultEffort: OFF_REASONING_EFFORT },
		});
	}
	async *stream(options) {
		const env = { stack: [], error: void 0, hasError: false };
		const dispose = async () => {
			while (env.stack.length) {
				const item = env.stack.pop();
				if (item && typeof item.dispose === "function") {
					try { await item.dispose(); } catch { /* teardown best-effort */ }
				}
			}
		};
		try {
			const connection = this.config.options();
			const apiKey = await this.config.resolveApiKey(connection);
			const userId = this.config.resolveUserId();
			// Image blocks may nest inside tool-result content (read_image's
			// output), which contentHasImage() does not see; detect recursively
			// and resolve the attachment service regardless.
			const hasImages = options.messages.some((message) => blocksHaveImage(message.content));
			const attachments = this.config.resolveAttachments?.();
			if (hasImages && attachments === void 0) {
				throw new LlmError("llm-mimo: image input requires the durable attachment service", "UNSUPPORTED_CONTENT");
			}
			const consumer = new AbortController();
			const watchdog = idleWatchdog(
				options.signal === void 0 ? consumer.signal : AbortSignal.any([options.signal, consumer.signal]),
				connection.streamIdleTimeoutMs,
				STREAM_IDLE_TIMEOUT_CODE,
			);
			const iterator = this.request(options, watchdog.signal, connection, apiKey, userId, attachments, () => watchdog.pulse())[Symbol.asyncIterator]();
			let exhausted = false;
			try {
				while (true) {
					const result = await watchdog.next(iterator);
					if (result.done) { exhausted = true; return; }
					yield result.value;
				}
			} catch (error) {
				if (timeoutOf(watchdog.signal, STREAM_IDLE_TIMEOUT_CODE) !== void 0) {
					throw new LlmError(`MiMo stream idle timeout after ${connection.streamIdleTimeoutMs}ms`, "TIMEOUT", { cause: error });
				}
				if (options.signal?.aborted) throw new LlmError("MiMo request aborted by caller", "ABORTED", { cause: error });
				if (error instanceof LlmError) throw error;
				throw new LlmError(`MiMo API stream from ${connection.baseURL} failed`, "TRANSPORT", { cause: error });
			} finally {
				consumer.abort("MiMo stream consumer stopped");
				if (!exhausted && iterator.return !== void 0) {
					try { await iterator.return(); } catch { /* transport teardown */ }
				}
			}
		} catch (e) {
			env.error = e;
			env.hasError = true;
		} finally {
			await dispose();
		}
		if (env.hasError) throw env.error;
	}
	async *request(options, signal, connection, apiKey, userId, attachments, onComment) {
		const body = await serializeRequest(options, connection.defaults ?? {}, attachments, connection);
		const payload = JSON.stringify(body);
		const headers = {
			"authorization": `Bearer ${apiKey}`,
			"api-key": apiKey,
			"content-type": "application/json",
			"accept": "text/event-stream",
			...attributionHeaders(),
			"x-dsh-user-id": String(userId),
			...options.sessionId !== void 0 ? { "x-dsh-session-id": String(options.sessionId) } : {},
		};
		let response;
		try {
			response = await fetch(`${connection.baseURL}/chat/completions`, {
				method: "POST",
				headers,
				body: payload,
				signal,
			});
		} catch (error) {
			if (signal.aborted) throw error;
			throw new LlmError(`MiMo API request to ${connection.baseURL} failed`, "TRANSPORT", { cause: error });
		}
		if (!response.ok) {
			let message = `MiMo API error (HTTP ${response.status})`;
			let providerError;
			try {
				providerError = (await response.json()).error;
				if (providerError?.message) message = providerError.message;
			} catch { /* keep generic message */ }
			const delay = providerRetryAfterMs(response.headers.get("retry-after"));
			const id = requestId(response.headers);
			throw new LlmError(message, httpErrorCode(response.status, providerError), {
				status: response.status,
				...delay === void 0 ? {} : { providerRetryAfterMs: delay },
				...id === void 0 ? {} : { requestId: id },
			});
		}
		if (!response.body) throw new LlmError("MiMo API returned no response body", "EMPTY_RESPONSE");
		yield* translate(parseSse(response.body, onComment));
	}
};

// ── plugin registration ──────────────────────────────────────────────────────
function apply(ctx, config) {
	let current = () => config;
	let lastRaw;
	let lastGood;
	const options = () => {
		const raw = current();
		if (raw === lastRaw && lastGood !== void 0) return lastGood;
		try {
			const next = resolveAdapterOptions(raw, launchEnvironmentOf(ctx));
			lastRaw = raw;
			lastGood = next;
			return next;
		} catch (error) {
			if (lastGood === void 0) throw error;
			lastRaw = raw;
			ctx.logger.error("llm-mimo: keeping the last good configuration after an invalid settings section");
			ctx.logger.error(error);
			return lastGood;
		}
	};
	options();
	const resolveApiKey = async (connection) => {
		const ref = connection.apiKeyEnv;
		const credentials = ctx.get("credentials");
		if (credentials !== void 0) {
			const hit = await credentials.resolve(ref);
			if (hit !== void 0) return assertUsableApiKey(hit.value, "llm-mimo", ref);
		} else {
			const ambient = launchEnvironmentOf(ctx).get(ref);
			if (ambient !== void 0 && ambient.value.length > 0) return assertUsableApiKey(ambient.value, "llm-mimo", ref);
		}
		throw new LlmError(
			`llm-mimo: no API key for provider route "${PROVIDER}"; store ${ref} through the credentials service (the web Models page writes it), or export ${ref} in the launching environment`,
			"MISSING_CREDENTIAL",
		);
	};
	let userId;
	const resolveUserId = () => userId ??= getOrCreateAnonymousUserId();
	const adapter = new MiMoAdapter({
		options,
		resolveApiKey,
		resolveUserId,
		resolveAttachments: () => ctx.get("attachments"),
	});
	ctx.llm.registerConfigurableProviders([{
		provider: PROVIDER,
		displayName: "Xiaomi MiMo",
		settingsNs: NS,
		settingsPath: [],
	}]);
	const registration = ctx.llm.registerAdapter([PROVIDER], adapter);
	let registeredPolicy = options().retryPolicy;
	const ensureRegistrationFacts = () => {
		const policy = options().retryPolicy;
		if (deepEqualJson(policy, registeredPolicy)) return;
		registration.replace([PROVIDER]);
		registeredPolicy = policy;
	};
	installSettingsSection(ctx, NS, Config, config, {
		setSource: (source) => { current = source; },
		onChange: ensureRegistrationFacts,
	});
}

export { Config, DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, DEFAULT_STREAM_IDLE_TIMEOUT_MS, MiMoAdapter, PROVIDER, PUBLIC_BASE_URL, apply, inject, name, resolveAdapterOptions };
