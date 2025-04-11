def test_app_entrypoint_imports_and_registers():
    import src.app  # Should not raise
    # Check that event handlers and commands are imported
    assert hasattr(src.app, "on_chat_start")
