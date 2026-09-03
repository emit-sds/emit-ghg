"""
Command-line interface entry points for EMIT GHG processing.

These functions wrap the deployment scripts to provide clean CLI commands.
"""

def ghg_process_cli():
    """Entry point for emit-ghg-process command."""
    from deploy import ghg_process
    ghg_process.main()

def bulk_cli():
    """Entry point for emit-ghg-bulk command."""
    from deploy import run_bulk_ghg
    run_bulk_ghg.main()

def plumeid_cli():
    """Entry point for emit-ghg-plumeid command."""
    from deploy import run_plumeid_ghg
    run_plumeid_ghg.main()
