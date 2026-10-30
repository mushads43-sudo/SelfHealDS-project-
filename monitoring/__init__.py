"""Prometheus observability layer for SelfHealDS.

This package wraps the existing backend entry points (assess, heal,
run_research_suite) without editing them, so every research result produced
by SelfHealDS is byte-identical with or without monitoring enabled.
"""
