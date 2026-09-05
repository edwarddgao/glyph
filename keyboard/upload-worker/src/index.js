// SwipeRacer upload endpoint.
//
//   POST /save            body: one JSON record (kind race | native | kbcapture | kbpick), ≤ 2 MB
//                         header: Authorization: Bearer <UPLOAD_TOKEN>
//                         -> stored at <kind>/<session>/<ts>-<rand>.json in R2; {"ok":true}
//   GET  /list?cursor=    header: Authorization: Bearer <ADMIN_TOKEN>
//                         -> {"keys":[{key,size,uploaded}], "cursor": ...}   (research/iphone/sync_race.py)
//   GET  /obj/<key>       header: Authorization: Bearer <ADMIN_TOKEN>   -> the record
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

    if (req.method === "GET" && url.pathname === "/") return json({ service: "swipe-upload", ok: true });
    return json({ error: "not found" }, 404);
  },
};

async function sha256(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
