/**
 * StegGate JavaScript Client SDK
 * ================================
 * Works in browsers (vanilla JS, React, Vue, etc.) and Node.js.
 *
 * Browser usage:
 *   <script src="steggate.js"></script>
 *   const client = new StegGate("http://localhost:5050");
 *
 * ESM / bundler usage:
 *   import { StegGate } from "./steggate.js";
 *
 * Node.js (with node-fetch or native fetch ≥18):
 *   const { StegGate } = require("./steggate.js");
 */

// ─────────────────────────────────────────────────────────────────────────────
//  ScanResult — returned by client.sanitize()
// ─────────────────────────────────────────────────────────────────────────────

class ScanResult {
  /**
   * @param {boolean} isThreat
   * @param {number}  riskScore       0–100
   * @param {boolean} wasSanitized
   * @param {Blob}    safeBlob        ready to display or upload
   * @param {string}  outputName      filename of clean file
   * @param {string}  originalName
   * @param {number}  scanDurationMs
   * @param {Headers} headers         raw response headers
   */
  constructor({ isThreat, riskScore, wasSanitized, safeBlob,
                outputName, originalName, scanDurationMs, headers }) {
    this.isThreat       = isThreat;
    this.riskScore      = riskScore;
    this.wasSanitized   = wasSanitized;
    this.safeBlob       = safeBlob;
    this.outputName     = outputName;
    this.originalName   = originalName;
    this.scanDurationMs = scanDurationMs;
    this.headers        = headers;
  }

  /** Risk level string: "NONE" | "LOW" | "MEDIUM" | "HIGH" */
  get threatLevel() {
    if (this.riskScore >= 75) return "HIGH";
    if (this.riskScore >= 50) return "MEDIUM";
    if (this.riskScore >= 25) return "LOW";
    return "NONE";
  }

  /** Object URL for use in <img src> or <a href> — remember to revoke when done */
  get objectURL() {
    if (this._url) return this._url;
    this._url = URL.createObjectURL(this.safeBlob);
    return this._url;
  }

  /** Revoke the object URL to free memory */
  revokeURL() {
    if (this._url) { URL.revokeObjectURL(this._url); this._url = null; }
  }

  /**
   * Trigger a browser download of the clean image.
   * @param {string} [filename]
   */
  download(filename) {
    const a = document.createElement("a");
    a.href     = this.objectURL;
    a.download = filename || this.outputName;
    a.click();
  }

  /**
   * Upload the sanitized image back to your own server.
   * @param {string} uploadUrl  - your site's upload endpoint
   * @param {string} fieldName  - form field name (default "file")
   * @param {Object} extra      - additional FormData fields
   */
  async uploadToServer(uploadUrl, fieldName = "file", extra = {}) {
    const form = new FormData();
    form.append(fieldName, this.safeBlob, this.outputName);
    for (const [k, v] of Object.entries(extra)) form.append(k, v);
    const r = await fetch(uploadUrl, { method: "POST", body: form });
    if (!r.ok) throw new Error(`Upload failed: ${r.status} ${r.statusText}`);
    return r;
  }

  toJSON() {
    return {
      isThreat: this.isThreat, riskScore: this.riskScore,
      wasSanitized: this.wasSanitized, threatLevel: this.threatLevel,
      outputName: this.outputName, originalName: this.originalName,
      scanDurationMs: this.scanDurationMs,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  StegGateError
// ─────────────────────────────────────────────────────────────────────────────

class StegGateError extends Error {
  constructor(statusCode, detail) {
    super(`StegGate ${statusCode}: ${detail}`);
    this.statusCode = statusCode;
    this.detail     = detail;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  StegGate — main client class
// ─────────────────────────────────────────────────────────────────────────────

class StegGate {
  /**
   * @param {string}   baseUrl   StegGate server URL
   * @param {Object}   options
   * @param {string}   options.apiKey       Bearer token (if server requires auth)
   * @param {boolean}  options.force        Always sanitize, even if clean
   * @param {string}   options.webhookUrl   Server will POST scan result here
   * @param {Function} options.onThreat     Callback(result) when threat detected
   * @param {Function} options.onProgress   Callback(phase: string) for UI updates
   */
  constructor(baseUrl, {
    apiKey     = "",
    force      = false,
    webhookUrl = "",
    onThreat   = null,
    onProgress = null,
  } = {}) {
    if (!baseUrl) throw new Error("baseUrl is required");
    this.baseUrl     = baseUrl.replace(/\/$/, "");
    this.apiKey      = apiKey;
    this.force       = force;
    this.webhookUrl  = webhookUrl;
    this.onThreat    = onThreat;
    this.onProgress  = onProgress;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Scan a File/Blob and return a ScanResult.
   *
   * @param   {File|Blob}  file
   * @param   {Object}     [options]   Override instance options for this call
   * @returns {Promise<ScanResult>}
   *
   * @example
   * const client = new StegGate("http://localhost:5050");
   *
   * input.addEventListener("change", async () => {
   *   const result = await client.sanitize(input.files[0]);
   *   img.src = result.objectURL;           // show clean preview
   *   await result.uploadToServer("/store"); // send to your server
   * });
   */
  async sanitize(file, options = {}) {
    const force      = options.force      ?? this.force;
    const webhookUrl = options.webhookUrl ?? this.webhookUrl;

    this._emit("scanning");

    const form = new FormData();
    form.append("file",        file, file.name || "upload");
    form.append("force",       String(force));
    form.append("webhook_url", webhookUrl);

    const headers = {};
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    let resp;
    try {
      resp = await fetch(`${this.baseUrl}/api/sanitize`, {
        method: "POST", body: form, headers,
      });
    } catch (err) {
      throw new StegGateError(0, `Network error: ${err.message}`);
    }

    if (!resp.ok) {
      let detail = resp.statusText;
      try { detail = (await resp.json()).detail || detail; } catch {}
      throw new StegGateError(resp.status, detail);
    }

    const blob = await resp.blob();
    const h    = resp.headers;

    const result = new ScanResult({
      isThreat:      h.get("x-threat-detected")  === "true",
      riskScore:     parseFloat(h.get("x-risk-score")       || "0"),
      wasSanitized:  h.get("x-was-sanitized")    === "true",
      safeBlob:      blob,
      outputName:    _parseFilename(h.get("content-disposition") || "", file.name),
      originalName:  h.get("x-original-filename") || file.name,
      scanDurationMs: parseInt(h.get("x-scan-duration-ms") || "0", 10),
      headers:       h,
    });

    if (result.isThreat && typeof this.onThreat === "function") {
      this.onThreat(result);
    }

    this._emit(result.isThreat ? "threat" : "clean");
    return result;
  }

  /**
   * Wire a <input type="file"> so uploads are automatically scanned.
   * Replaces the file in the input with the sanitized version.
   *
   * @param {HTMLInputElement}   input
   * @param {HTMLImageElement}   [preview]   Optional <img> to update
   * @param {Function}           [callback]  (result: ScanResult) => void
   *
   * @example
   * const client = new StegGate("http://localhost:5050");
   * client.watchInput(
   *   document.querySelector("#avatar"),
   *   document.querySelector("#preview"),
   *   (result) => {
   *     statusEl.textContent = result.wasSanitized
   *       ? `⚠ Stego detected — cleaned (risk ${result.riskScore}%)`
   *       : `✓ Image is clean`;
   *   }
   * );
   */
  watchInput(input, preview = null, callback = null) {
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      if (!file) return;

      try {
        const result = await this.sanitize(file);

        // Replace the file in the input with the clean version
        const dt = new DataTransfer();
        dt.items.add(new File([result.safeBlob], result.outputName,
                              { type: result.safeBlob.type }));
        input.files = dt.files;

        // Update preview
        if (preview) {
          if (preview._stegUrl) URL.revokeObjectURL(preview._stegUrl);
          preview._stegUrl = result.objectURL;
          preview.src = preview._stegUrl;
        }

        if (typeof callback === "function") callback(result);
      } catch (err) {
        console.error("[StegGate]", err);
        if (typeof callback === "function") callback(null, err);
      }
    });
    return this;  // chainable
  }

  /**
   * Intercept a <form> submission — scan all image fields before submitting.
   *
   * @param {HTMLFormElement}  form
   * @param {string[]}         [fields]   Names of file inputs to scan (default: all)
   * @param {Function}         [callback] (results: ScanResult[]) => void
   */
  watchForm(form, fields = null, callback = null) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const inputs = fields
        ? fields.map(n => form.elements[n]).filter(Boolean)
        : [...form.querySelectorAll('input[type="file"]')];

      const results = [];
      for (const input of inputs) {
        const file = input.files?.[0];
        if (!file) continue;
        const result = await this.sanitize(file);
        const dt = new DataTransfer();
        dt.items.add(new File([result.safeBlob], result.outputName,
                              { type: result.safeBlob.type }));
        input.files = dt.files;
        results.push(result);
      }

      if (typeof callback === "function") await callback(results);
      form.submit();
    });
    return this;
  }

  /** Check server health */
  async health() {
    const headers = this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {};
    const r = await fetch(`${this.baseUrl}/api/health`, { headers });
    return r.json();
  }

  _emit(phase) {
    if (typeof this.onProgress === "function") this.onProgress(phase);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────────────────────────────────────

function _parseFilename(disposition, fallback) {
  const m = disposition.match(/filename="?([^";\r\n]+)"?/);
  return m ? m[1] : fallback;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Exports — works as ESM, CJS, and browser global
// ─────────────────────────────────────────────────────────────────────────────

if (typeof module !== "undefined" && module.exports) {
  module.exports = { StegGate, ScanResult, StegGateError };                 // CJS
} else if (typeof define === "function" && define.amd) {
  define([], () => ({ StegGate, ScanResult, StegGateError }));              // AMD
} else if (typeof window !== "undefined") {
  window.StegGate      = StegGate;                                          // Browser global
  window.ScanResult    = ScanResult;
  window.StegGateError = StegGateError;
}

export { StegGate, ScanResult, StegGateError };                             // ESM