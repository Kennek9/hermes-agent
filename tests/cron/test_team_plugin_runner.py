import pytest
from hermes_cli import plugins
from cron import jobs,scheduler

@pytest.fixture
def manager(monkeypatch):
    value=plugins.PluginManager()
    monkeypatch.setattr(plugins,"get_plugin_manager",lambda:value)
    monkeypatch.setattr(plugins,"discover_plugins",lambda:None)
    return value

def test_missing_runner_never_falls_back_to_agent(manager):
    job={"id":"test","prompt":"must not run","plugin_runner":"missing"}
    assert scheduler.run_job(job)[3]=="plugin_runner_unavailable"
    ctx=plugins.PluginContext(plugins.PluginManifest(name="team_test",source="user"),manager)
    def callback(*,job,cancel_event=None):
        job["plugin_data"]["task_id"]="mutated-copy"
        return True,"audit only","",None
    handle=ctx.register_cron_job_runner("team",callback)
    job.update(plugin_runner="team",plugin_data={"task_id":"bound"})
    assert scheduler.run_job(job)==(True,"audit only","",None)
    assert job["plugin_data"]["task_id"]=="bound"
    handle.dispose()
    assert scheduler.run_job(job)[3]=="plugin_runner_unavailable"

def test_runner_failure_is_bounded(manager):
    ctx=plugins.PluginContext(plugins.PluginManifest(name="team_test",source="user"),manager)
    def callback(**kwargs): raise ValueError("private-data")
    ctx.register_cron_job_runner("team",callback)
    result=scheduler.run_job({"id":"test","plugin_runner":"team"})
    assert result[3]=="plugin_runner_failed" and "private-data" not in str(result)

def test_disabled_creation_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job=jobs.create_job(prompt="read-only",schedule="every 5m",plugin_runner="team",plugin_data={"task_id":"one","revision":1},enabled=False,deliver="local")
    assert job["state"]=="paused" and not job["enabled"]
    assert not jobs.claim_job_for_fire(job["id"])
    assert jobs.get_job(job["id"])["plugin_data"]=={"task_id":"one","revision":1}

@pytest.mark.parametrize("base,expr,zone,expected",[
    ("2026-09-28T00:00:00+00:00","0 9 * * *","Asia/Shanghai","2026-09-28T09:00:00+08:00"),
    ("2026-03-07T02:30:00-05:00","30 2 * * *","America/New_York","2026-03-09T02:30:00-04:00"),
    ("2026-11-01T01:30:00-04:00","30 1 * * *","America/New_York","2026-11-02T01:30:00-05:00"),
])
def test_explicit_timezone_gap_and_fold(base,expr,zone,expected):
    assert jobs.compute_next_run({"kind":"cron","expr":expr,"timezone":zone},base)==expected
