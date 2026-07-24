from em_volume_tools.manifest import Manifest


def test_record_load_roundtrip(tmp_path):
    p = str(tmp_path / "prog.jsonl")
    m = Manifest(p)
    m.reset()
    m.record(0, [((0, 0, 0), "written"), ((0, 0, 1), "empty")])
    m.record(1, [((0, 0, 0), "written")])
    m.close()

    m2 = Manifest(p).load()
    assert m2.is_done(0, (0, 0, 0))
    assert m2.is_done(0, (0, 0, 1))       # empty counts as done
    assert m2.is_done(1, (0, 0, 0))
    assert not m2.is_done(0, (9, 9, 9))
    assert m2.done_indices(0) == {(0, 0, 0), (0, 0, 1)}
    assert m2.counts() == {"written": 2, "empty": 1}


def test_reset_truncates(tmp_path):
    p = str(tmp_path / "prog.jsonl")
    m = Manifest(p)
    m.reset()
    m.record(0, [((0, 0, 0), "written")])
    m.close()
    Manifest(p).reset()                    # fresh run wipes prior records
    assert Manifest(p).load().done_indices(0) == set()


def test_tolerates_torn_final_line(tmp_path):
    p = str(tmp_path / "prog.jsonl")
    with open(p, "w") as f:
        f.write('{"level": 0, "index": [0, 0, 0], "status": "written"}\n')
        f.write('{"level": 0, "index": [0, 0, 1], "sta')   # crash mid-write
    m = Manifest(p).load()
    assert m.is_done(0, (0, 0, 0))
    assert not m.is_done(0, (0, 0, 1))     # torn line ignored, not fatal


def test_no_path_is_in_memory_only(tmp_path):
    m = Manifest(None)
    m.reset()
    m.record(0, [((1, 2, 3), "written")])
    assert m.is_done(0, (1, 2, 3))
