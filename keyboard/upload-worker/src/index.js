// SwipeRacer upload endpoint.
//
//   POST /save            body: one JSON record (kind race | native | kbcapture | kbpick), ≤ 2 MB
//                         header: Authorization: Bearer <UPLOAD_TOKEN>
//                         -> stored at <kind>/<session>/<ts>-<rand>.json in R2; {"ok":true}
//   GET  /list?cursor=    header: Authorization: Bearer <ADMIN_TOKEN>
//                         -> {"keys":[{key,size,uploaded}], "cursor": ...}   (research/iphone/sync_race.py)
//   GET  /obj/<key>       header: Authorization: Bearer <ADMIN_TOKEN>   -> the record
//   GET  /privacy         the app's privacy policy (linked from the app and App Store Connect)
//
// The upload token lives in the app bundle, so it only stops drive-by writes;
// the rate limit and size cap bound what a leaked token can do, and every
// record is prompted-word gesture data under a random per-install id.
const KINDS = new Set(["race", "native", "kbcapture", "kbpick"]);
const MAX_BYTES = 2_000_000;

function bearer(req) {
  const h = req.headers.get("Authorization") || "";
  return h.startsWith("Bearer ") ? h.slice(7).trim() : "";
}
function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
}
function safe(s, n) {
  return String(s || "anon").toLowerCase().replace(/[^a-z0-9_-]/g, "").slice(0, n) || "anon";
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);

    if (req.method === "POST" && url.pathname === "/save") {
      if (bearer(req) !== env.UPLOAD_TOKEN) return json({ error: "unauthorized" }, 401);
      const ip = req.headers.get("CF-Connecting-IP") || "0";
      if (env.RATE) {
        const { success } = await env.RATE.limit({ key: ip });
        if (!success) return json({ error: "rate limited" }, 429);
      }
      const len = Number(req.headers.get("Content-Length") || 0);
      if (len > MAX_BYTES) return json({ error: "too large" }, 413);
      const text = await req.text();
      if (text.length > MAX_BYTES) return json({ error: "too large" }, 413);
      let rec;
      try { rec = JSON.parse(text); } catch { return json({ error: "bad json" }, 400); }
      if (!rec || typeof rec !== "object" || !KINDS.has(rec.kind)) return json({ error: "bad kind" }, 400);
      const ts = Number.isFinite(rec.ts) ? Math.trunc(rec.ts) : Date.now();
      const key = `${rec.kind}/${safe(rec.session, 32)}/${ts}-${crypto.randomUUID().slice(0, 8)}.json`;
      await env.RACES.put(key, text, {
        httpMetadata: { contentType: "application/json" },
        customMetadata: { ip_hash: await sha256(ip).then((h) => h.slice(0, 16)), ua: (req.headers.get("User-Agent") || "").slice(0, 80) },
      });
      return json({ ok: true, key });
    }

    if (req.method === "GET" && url.pathname === "/list") {
      if (bearer(req) !== env.ADMIN_TOKEN) return json({ error: "unauthorized" }, 401);
      const r = await env.RACES.list({ cursor: url.searchParams.get("cursor") || undefined, prefix: url.searchParams.get("prefix") || undefined, limit: 1000 });
      return json({ keys: r.objects.map((o) => ({ key: o.key, size: o.size, uploaded: o.uploaded })), cursor: r.truncated ? r.cursor : null });
    }

    if (req.method === "GET" && url.pathname.startsWith("/obj/")) {
      if (bearer(req) !== env.ADMIN_TOKEN) return json({ error: "unauthorized" }, 401);
      const obj = await env.RACES.get(decodeURIComponent(url.pathname.slice(5)));
      if (!obj) return json({ error: "not found" }, 404);
      return new Response(obj.body, { headers: { "Content-Type": "application/json" } });
    }

    if (req.method === "GET" && url.pathname === "/privacy") {
      return new Response(PRIVACY, { headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "public, max-age=3600" } });
    }
    if (req.method === "GET" && url.pathname === "/") return json({ service: "swipe-upload", ok: true });
    return json({ error: "not found" }, 404);
  },
};

async function sha256(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const PRIVACY = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glyph privacy policy</title>
<style>body{font:17px/1.55 -apple-system,system-ui,sans-serif;max-width:40em;margin:3em auto;padding:0 1.2em;color:#111}h1{font-size:1.6em}h2{font-size:1.15em;margin-top:2em}code{background:#f2f2f4;padding:.1em .3em;border-radius:4px}@media(prefers-color-scheme:dark){body{background:#111;color:#eee}code{background:#222}}</style></head><body>
<h1>Glyph privacy policy</h1>
<p><em>Glyph</em> is an open-source swipe keyboard for iPhone. This page says exactly what the app records, what it never records, and what happens to the data. Last updated 5 September 2026.</p>

<h2>The keyboard records nothing</h2>
<p>The Glyph keyboard extension has <strong>no network access</strong> and does not request Full Access. Nothing you type with it — in any app — is stored, sent anywhere, or read by us. Decoding runs entirely on your phone.</p>

<h2>What Practice records</h2>
<p>The app's Practice mode shows you sentences and asks you to swipe their words on a keyboard drawn inside the app. Playing it uploads, for each prompted word:</p>
<ul>
<li>the finger path on that in-app keyboard: x, y coordinates and timestamps;</li>
<li>the word you were asked to swipe, the sentence it came from, and whether the swipe passed the geometric check;</li>
<li>what the on-device decoder read, so the model's mistakes can be studied;</li>
<li>a random install id (a short string generated on first practice, shown under the app's details screen), the app build number, and the sentence timing.</li>
</ul>
<p>The upload server also stores a truncated hash of the uploading IP address, used only for rate limiting, and the request's User-Agent string (iOS version). No name, email, Apple ID, device identifier, location, contacts, or anything else from your phone is collected. There are no accounts, analytics SDKs or advertising.</p>

<h2>What the data is for</h2>
<p>Training and evaluating the open-source swipe decoder. The recorded gestures with their prompted words may be published as an open research dataset under the same random ids. The code, models and research notebook are at <a href="https://github.com/edwarddgao/glyph">github.com/edwarddgao/glyph</a>.</p>

<h2>Consent and deletion</h2>
<p>Practicing is the consent: the practice screen says what is recorded before you start, and there is no other way for the app to send anything. To have your records deleted, open an issue at <a href="https://github.com/edwarddgao/glyph/issues">github.com/edwarddgao/glyph/issues</a> quoting the install id from the app's details screen (or delete and reinstall the app, which discards the id on your side). Records are otherwise kept as research data.</p>

<h2>Children</h2>
<p>Glyph is not directed at children under 13 and collects no personal information from anyone.</p>
</body></html>`;
