#!/usr/bin/env node
/**
 * dsh-llm-mimo demo — read one image with Xiaomi MiMo (mimo-v2.5) directly
 * through the official OpenAI-compatible API, without the harness.
 *
 * Usage:  MIMO_API_KEY=sk-xxxx node demo/read-image.mjs <image-path> [prompt]
 * Example: node demo/read-image.mjs ./chart.png "逐字抄录图中的数字与文字"
 */
import { readFile } from "node:fs/promises";
import { extname } from "node:path";

const key = process.env.MIMO_API_KEY;
if (!key) {
  console.error("MIMO_API_KEY is not set");
  process.exit(1);
}
const imagePath = process.argv[2];
if (!imagePath) {
  console.error("usage: MIMO_API_KEY=sk-xxx node demo/read-image.mjs <image-path> [prompt]");
  process.exit(1);
}
const prompt = process.argv[3] ?? "请描述这张图片的内容。";

const mediaType = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
}[extname(imagePath).toLowerCase()];
if (!mediaType) {
  console.error(`unsupported image type: ${extname(imagePath)} (use PNG/JPEG/WebP/GIF)`);
  process.exit(1);
}

const base64 = (await readFile(imagePath)).toString("base64");
const response = await fetch("https://api.xiaomimimo.com/v1/chat/completions", {
  method: "POST",
  headers: {
    authorization: `Bearer ${key}`,
    "api-key": key,
    "content-type": "application/json",
  },
  body: JSON.stringify({
    model: "mimo-v2.5",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: `data:${mediaType};base64,${base64}` } },
        ],
      },
    ],
    max_tokens: 4096,
  }),
});
if (!response.ok) {
  console.error(`MiMo API error (HTTP ${response.status}):`, await response.text());
  process.exit(1);
}
const data = await response.json();
const message = data.choices?.[0]?.message;
console.log(message?.reasoning_content ? `[reasoning]\n${message.reasoning_content}\n` : "");
console.log(message?.content ?? "(empty response)");
