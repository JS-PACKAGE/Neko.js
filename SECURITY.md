# Security

## Reporting a vulnerability

Please do not publish exploit details or sensitive data in a public issue. Report suspected vulnerabilities through GitHub's private vulnerability reporting for this repository when available. If it is unavailable, use the maintainer contact route listed on the repository profile and request a private disclosure channel. Include affected version, reproduction steps, impact, and any relevant mitigations. Do not include real user images, prompts, access tokens, or private model/cache contents.

## Threat model

Neko.js runs a pinned multimodal model locally in Node.js or a browser, and includes helpers for extracting remote HTML and image metadata. It is a prototype, not a security boundary. Model-generated text can be false, manipulated, or unsafe; never use it to make authorization, security, or other consequential decisions.

- **Untrusted page content:** HTML is parsed as data; scripts and page subresources are not executed or fetched by `extractPage`. This does not prevent prompt injection in extracted text or model output. Treat extracted content and all generated output as untrusted. `renderMarkdown` escapes report text, but applications must use a safe Markdown renderer and appropriate content security policy.
- **URL fetching / SSRF:** `extractPage` and `loadImage` accept HTTP(S) URLs. In Node.js, calling these APIs on attacker-controlled URLs can make requests from the application's network context, including to private or internal hosts. Do not expose arbitrary URL extraction as a server-side proxy; validate/allowlist destinations and enforce outbound network controls at the application boundary. Redirects and response-size/time limits do not replace SSRF defenses. Browser requests remain subject to browser same-origin and CORS rules.
- **Model downloads and cache:** Model assets are pinned to a Hub revision and checked against expected file sizes and SHA-256 before use or caching. Integrity failures are errors. These checks detect altered files relative to the checked-in manifest; they do not authenticate a compromised application build or replace review of model provenance and license. The browser Cache API is managed by the browser. Protect Node cache directories according to the host's user and filesystem access controls.
- **Local image processing:** The browser image helper accepts a limited set of raster types, checks signatures, caps input bytes and decoded dimensions, and rejects SVG. It still relies on browser image decoders and must not be treated as a general-purpose sanitizer. Prefer user-selected files when possible; remote image URLs are subject to CORS.
- **Third-party runtime:** Transformers.js, ONNX Runtime, WebAssembly and WebGPU drivers are part of the trusted computing base. Build and install with the committed lockfile, review dependency updates, and do not load the browser demo or package bundle into an untrusted page. A configured ONNX execution provider is not proof that every model operation runs on the corresponding device.

## Data handling

The project contains no application telemetry or model-output upload feature. Initial model setup downloads pinned assets from Hugging Face. Browser inference accepts local `Blob`/`File` inputs and runs in the browser; callers who pass remote URLs initiate network access subject to their runtime's policies. Applications remain responsible for their own logging, analytics, hosting, and privacy notices. Avoid logging prompts, image data, or generated content unless required and appropriately protected.

## Dependency and model integrity policy

Direct dependencies are version-pinned in `package.json`, and transitive versions are recorded in `package-lock.json`. Changes to the Transformers.js version, ONNX Runtime, model revision, pinned file sizes/digests, provider configuration, or cache implementation require review and targeted tests. Never weaken integrity failures into silent cache misses or use an unpinned model fallback.
The npm `allowScripts` list is intentionally limited to the pinned `esbuild`, `onnxruntime-node`, and `sharp` packages: their install hooks select or provide platform-specific build/runtime artifacts. Do not replace this with a blanket script approval; review any proposed change to the allowlist and the exact package version first.
