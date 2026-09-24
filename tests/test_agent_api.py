"""Agent + API end to end, with a scripted fake LLM (real Postgres, no model calls)."""

import json

import pytest
from conftest import needs_db
from fakes import FakeProvider
from fastapi.testclient import TestClient

from querynest import knowledge
from querynest.agent import Agent
from querynest.api import app
from querynest.api.deps import llm_router
from querynest.config import settings
from querynest.llm import LLMRouter
from querynest.masking import Masker
from querynest.permissions import system_user

pytestmark = needs_db
TOP3 = ("SELECT customername, SUM(salesamt) AS revenue FROM sales.invoices WHERE cyyyy = 2025 "
        "GROUP BY customername ORDER BY revenue DESC LIMIT 3")


def script():
    return [("tool", "run_sql_query", {"sql": TOP3, "title": "Top 3"}),
            ("tool", "create_chart", {"result_id": "__LAST__"}),
            ("text", "The top customer is [P1].")]


class ResultIdFake(FakeProvider):
    """Fills in the real result_id from the previous tool result, like a real model would."""

    def stream(self, system, messages, tools, on_text):
        if self.steps and self.steps[0][0] == "tool" and self.steps[0][2].get("result_id") == "__LAST__":
            last = json.loads(messages[-1]["content"])
            self.steps[0] = ("tool", self.steps[0][1], {"result_id": last["result_id"]})
        return super().stream(system, messages, tools, on_text)


def test_agent_masks_for_cloud_and_unmasks_for_user():
    fake = ResultIdFake(script())
    agent = Agent(system_user("manager"), LLMRouter([fake], use_cache=False), Masker())
    out = agent.run("Who were the top customers in 2025?")
    sent = json.dumps([m for r in fake.requests for m in r["messages"]], default=str)
    assert "Customer 0352" not in sent and "[P1]" in sent  # the cloud model only saw tokens
    assert "Customer 0352" in out["text"]  # the user sees the real name
    assert [b["kind"] for b in out["blocks"]] == ["result", "chart"]
    assert out["sql"] and out["example_id"]  # remembered as a learned example


def test_memory_never_stores_another_users_row_filter():
    from querynest.appdb import SqlExample, session
    from querynest.permissions import UserContext

    sales = UserContext(username="t-sales", role="sales", attributes={"salesman": "Salesman 12"})
    sql = "SELECT customername, SUM(salesamt) AS revenue FROM sales.invoices GROUP BY 1 ORDER BY 2 DESC LIMIT 3"
    fake = FakeProvider([("tool", "run_sql_query", {"sql": sql}), ("text", "done")])
    out = Agent(sales, LLMRouter([fake], use_cache=False)).run("My top 3 customers overall?")
    with session() as s:
        stored = s.get(SqlExample, out["example_id"]).sql
    assert stored == sql and "Salesman 12" not in stored


def test_local_model_gets_real_data():
    fake = ResultIdFake(script(), is_local=True)
    Agent(system_user("manager"), LLMRouter([fake], use_cache=False)).run("Top customers 2025?")
    assert "Customer 0352" in json.dumps([m for r in fake.requests for m in r["messages"]], default=str)


def test_knowledge_retrieval_finds_terms_and_examples():
    found = knowledge.retrieve("Who are the top customers by revenue in 2025?")
    assert {"revenue", "customer"} <= {t.term for t in found.terms}
    assert any("Top 10 customers" in ex.question for ex, _ in found.examples)


# ------------------------------------------------------------------ API
@pytest.fixture
def client():
    fake = ResultIdFake(script())
    app.dependency_overrides[llm_router] = lambda: LLMRouter([fake], use_cache=False)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def login(c, username):
    r = c.post("/api/auth/login", json={"username": username, "password": settings.demo_password.get_secret_value()})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_full_chat_flow(client):
    h = login(client, "manager")
    conv = client.post("/api/conversations", headers=h).json()
    r = client.post(f"/api/conversations/{conv['id']}/messages", headers=h, json={"question": "Top customers 2025?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    assert any(e["type"] == "text" for e in events)
    done = events[-1]
    result_id = done["blocks"][0]["result_id"]

    full = client.get(f"/api/results/{result_id}", headers=h).json()
    assert full["row_count"] == 3
    xlsx = client.get(f"/api/results/{result_id}/excel", headers=h)
    assert xlsx.status_code == 200 and xlsx.content[:2] == b"PK"  # .xlsx files are zip archives

    detail = client.get(f"/api/conversations/{conv['id']}", headers=h).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert client.post(f"/api/messages/{done['message_id']}/feedback", headers=h, json={"helpful": True}).status_code == 204

    hs = login(client, "sales12")  # other users can't see it
    assert client.get(f"/api/conversations/{conv['id']}", headers=hs).status_code == 404
    assert client.get(f"/api/results/{result_id}", headers=hs).status_code == 403
    client.delete(f"/api/conversations/{conv['id']}", headers=h)


def test_auth_and_admin_protection(client):
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", headers={"Authorization": "Bearer forged"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
    assert client.get("/api/admin/audit", headers=login(client, "sales12")).status_code == 403
    admin = login(client, "admin")
    assert client.get("/api/admin/audit", headers=admin).status_code == 200
    bad = client.post("/api/admin/examples", headers=admin, json={"question": "x", "sql": "DELETE FROM sales.invoices"})
    assert bad.status_code == 422  # curated examples must pass the guardrails too
