"""
StegGate Admin CLI
==================
Manage API keys and view usage from the command line.

Usage:
    python admin.py create  --name "Acme Corp"  --email "dev@acme.com"
    python admin.py list
    python admin.py revoke  sg_abc123...
    python admin.py usage
    python admin.py usage   --key sg_abc123...  --days 30
"""

import argparse, os, secrets, sqlite3, sys, time
from pathlib import Path

DB_PATH = os.environ.get("DB_PATH", "steggate.db")

def _db():
    if not Path(DB_PATH).exists():
        print(f"[error] Database not found: {DB_PATH}")
        print("        Run server.py first to initialise the database.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def cmd_create(args):
    key = "sg_" + secrets.token_urlsafe(32)
    with _db() as conn:
        conn.execute("""
            INSERT INTO api_keys (key, name, email, created_at, active, rate_limit, note)
            VALUES (?,?,?,?,1,?,?)
        """, (key, args.name, args.email or "", int(time.time()),
              args.rate_limit, args.note or ""))
    print(f"\n  ✓ API key created\n")
    print(f"  Name       : {args.name}")
    print(f"  Email      : {args.email or '—'}")
    print(f"  Rate limit : {args.rate_limit} req/min")
    print(f"\n  API Key    : {key}")
    print(f"\n  ⚠  Save this key now — it cannot be retrieved later.\n")

def cmd_list(args):
    with _db() as conn:
        rows = conn.execute(
            "SELECT key, name, email, created_at, active, rate_limit, note "
            "FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
    if not rows:
        print("No API keys found.")
        return
    print(f"\n  {'KEY PREFIX':<20}  {'NAME':<20}  {'EMAIL':<25}  {'ACTIVE':<8}  {'RATE'}")
    print(f"  {'─'*20}  {'─'*20}  {'─'*25}  {'─'*8}  {'─'*10}")
    for r in rows:
        active = "yes" if r["active"] else "REVOKED"
        print(f"  {r['key'][:20]:<20}  {r['name']:<20}  {(r['email'] or '—'):<25}  {active:<8}  {r['rate_limit']}/min")
    print()

def cmd_revoke(args):
    with _db() as conn:
        result = conn.execute(
            "UPDATE api_keys SET active=0 WHERE key LIKE ?", (args.key + "%",)
        )
        if result.rowcount == 0:
            print(f"[error] No key found matching: {args.key}")
            sys.exit(1)
    print(f"  ✓ Revoked key: {args.key}…")

def cmd_usage(args):
    days  = args.days
    since = int(time.time()) - (days * 86400)
    with _db() as conn:
        if args.key:
            rows = conn.execute("""
                SELECT ts, filename, file_bytes, is_threat, risk_score,
                       was_sanitized, scan_ms, upload_id, ip
                FROM usage_log WHERE api_key LIKE ? AND ts >= ?
                ORDER BY ts DESC LIMIT 50
            """, (args.key + "%", since)).fetchall()
            print(f"\n  Last {days} days — key prefix: {args.key}\n")
            print(f"  {'TIME':<12}  {'FILE':<30}  {'THREAT':<8}  {'RISK':<8}  {'SANITIZED':<10}  {'MS'}")
            print(f"  {'─'*12}  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*6}")
            for r in rows:
                ts  = time.strftime("%m-%d %H:%M", time.localtime(r["ts"]))
                thr = "YES" if r["is_threat"] else "no"
                san = "YES" if r["was_sanitized"] else "no"
                print(f"  {ts:<12}  {(r['filename'] or '—')[:30]:<30}  {thr:<8}  {r['risk_score']:<8.1f}  {san:<10}  {r['scan_ms']}")
        else:
            rows = conn.execute("""
                SELECT u.api_key, k.name,
                    COUNT(*)              AS reqs,
                    SUM(u.is_threat)      AS threats,
                    SUM(u.was_sanitized)  AS sanitized,
                    ROUND(SUM(u.file_bytes)/1024.0/1024.0, 1) AS total_mb,
                    ROUND(AVG(u.scan_ms)) AS avg_ms
                FROM usage_log u
                LEFT JOIN api_keys k ON k.key = u.api_key
                WHERE u.ts >= ?
                GROUP BY u.api_key
                ORDER BY reqs DESC
            """, (since,)).fetchall()
            print(f"\n  Usage — last {days} days\n")
            print(f"  {'NAME':<22}  {'REQUESTS':<10}  {'THREATS':<10}  {'SANITIZED':<12}  {'MB':<8}  {'AVG MS'}")
            print(f"  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}")
            for r in rows:
                name = r["name"] or r["api_key"][:16]
                print(f"  {name:<22}  {r['reqs']:<10}  {r['threats']:<10}  {r['sanitized']:<12}  {r['total_mb']:<8}  {r['avg_ms']}")
    print()

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="admin.py", description="StegGate Admin CLI")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create", help="Issue a new API key")
    p.add_argument("--name",       required=True, help="Customer / project name")
    p.add_argument("--email",      default="",    help="Customer email")
    p.add_argument("--rate-limit", type=int, default=60, dest="rate_limit")
    p.add_argument("--note",       default="")

    sub.add_parser("list", help="List all API keys")

    p = sub.add_parser("revoke", help="Revoke a key by prefix")
    p.add_argument("key", help="First 16+ characters of the key to revoke")

    p = sub.add_parser("usage", help="View usage stats")
    p.add_argument("--key",  default="", help="Filter by key prefix")
    p.add_argument("--days", type=int, default=7)

    args = parser.parse_args()
    {"create": cmd_create, "list": cmd_list,
     "revoke": cmd_revoke, "usage": cmd_usage}[args.cmd](args)