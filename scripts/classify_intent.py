#!/usr/bin/env python3
"""
Intent-based phase classifier for SWE-Agent trajectories.

Classifies each step into a workflow phase based on what the agent
is TRYING TO DO, not which tool it called.

Phases:
  O = ORIENT    — discovering repo structure, finding relevant files
  R = READ      — reading/understanding specific source code  
  D = REPRODUCE — confirming the bug exists (pre-implementation)
  I = IMPLEMENT — making code changes to source files
  V = VERIFY    — confirming the fix works (post-implementation)
  S = SUBMIT    — submitting the patch
  ! = ERROR     — failed/broken command, empty action
"""

import json
import re
import os


def _is_directory_target(action_line):
    """Check if a view command targets a directory (not a file)."""
    # Extract the path from 'str_replace_editor view /some/path'
    parts = action_line.split()
    if len(parts) < 3:
        return False
    path = parts[2]
    # Directories: no file extension, or known dir patterns
    if '--view_range' in action_line:
        return False  # ranged view is always a file
    basename = path.split('/')[-1]
    # Has a file extension → file
    if '.' in basename and not basename.startswith('.'):
        return False
    return True


def _extract_bash_command(action):
    """Extract the actual command from bash -lc wrapping."""
    a = action.strip()
    # Unwrap bash -lc "..." or bash -lc '...'
    m = re.match(r'bash\s+-lc\s+["\'](.+)', a, re.DOTALL)
    if m:
        return m.group(1).rstrip("\"'}")
    # cd /app && ... 
    if a.startswith('cd ') and '&&' in a:
        return a.split('&&', 1)[1].strip()
    return a


def _is_test_command(cmd):
    """Check if a bash command is running a test suite."""
    cl = cmd.lower()
    return any(kw in cl for kw in [
        'pytest', 'python -m pytest', 'go test', 'npm test',
        'npx jest', 'mocha', 'python -m unittest',
    ])


def _is_search_command(cmd):
    """Check if a bash command is searching/exploring."""
    cl = cmd.lower()
    return any(cl.lstrip().startswith(p) for p in [
        'grep ', 'find ', 'rg ', 'ag ', 'ls ', 'ls\n',
        'tree ', 'wc ', 'file ',
    ]) or ('grep ' in cl and '|' in cl)


def _is_read_command(cmd):
    """Check if a bash command is reading file contents."""
    cl = cmd.lower().strip()
    return any(cl.startswith(p) for p in [
        'cat ', 'head ', 'tail ', 'sed -n', 'nl ', 'awk ',
    ])


def _is_run_command(cmd):
    """Check if a bash command runs a script/snippet."""
    cl = cmd.lower().strip()
    return any(cl.startswith(p) for p in [
        'python ', 'python3 ', 'node ', 'go run',
        'python -', 'python3 -',
    ])


def _is_compile_check(cmd):
    """Check if a bash command is a syntax/compile check."""
    cl = cmd.lower()
    return any(kw in cl for kw in [
        'py_compile', 'compileall', 'node -c ',
        'go build', 'go vet', 'make ',
    ])


def _is_error_obs(obs):
    """Check if the observation indicates the command failed/errored."""
    if not obs:
        return False
    o = obs[:500].lower()
    return any(kw in o for kw in [
        'syntax error', 'unexpected token', 'command not found',
        "here-document at line", "unexpected `}'",
        'invalid number of lines',
        'invalid option',
    ])


def _get_run_target(cmd):
    """Extract the script name from a run command."""
    # python /app/repro.py → repro.py
    # python test_edge_cases.py → test_edge_cases.py
    cl = cmd.strip()
    parts = cl.split()
    for p in parts[1:]:
        if p.endswith('.py') or p.endswith('.js') or p.endswith('.go'):
            return p.split('/')[-1].lower()
        if p.startswith('-'):
            continue
        break
    return ''


def _is_repro_filename(name):
    return any(kw in name for kw in ['repro', 'reproduce', 'demo'])


def _is_test_filename(name):
    return any(kw in name for kw in [
        'test_', '_test.', 'test.py', 'test.js', 'test.go',
        'verify', 'check', 'validate', 'edge_case',
    ])


def _is_doc_filename(name):
    return any(kw in name for kw in [
        'summary', 'readme', 'changes', 'implementation',
    ])


def classify_trajectory(trajectory):
    """
    Classify each step in a SWE-Agent trajectory into intent phases.
    
    Args:
        trajectory: list of step dicts with 'action' and 'observation' keys
    
    Returns:
        list of phase labels (one per step)
    """
    # First pass: find the index of the first source-code edit
    first_edit_idx = None
    edited_files = set()
    
    for i, step in enumerate(trajectory):
        action = step['action'].strip()
        fl = action.split('\n')[0].lower()
        
        if (fl.startswith('str_replace_editor str_replace') or 
            fl.startswith('str_replace_editor insert') or
            'applypatch' in fl):
            # Extract target file
            parts = action.split('\n')[0].split()
            if fl.startswith('str_replace_editor'):
                if len(parts) >= 3:
                    target = parts[2]
                    # Skip if it's editing a repro/test file the agent created
                    fname = target.split('/')[-1].lower()
                    if _is_repro_filename(fname) or _is_test_filename(fname) or _is_doc_filename(fname):
                        continue
                    if first_edit_idx is None:
                        first_edit_idx = i
                    edited_files.add(target)
            else:
                # applypatch
                if first_edit_idx is None:
                    first_edit_idx = i
    
    # If no edit found, everything is pre-implementation
    if first_edit_idx is None:
        first_edit_idx = len(trajectory)
    
    # Second pass: classify each step
    labels = []
    for i, step in enumerate(trajectory):
        action = step['action'].strip()
        obs = step.get('observation', '') or ''
        fl = action.split('\n')[0]
        fl_lower = fl.lower()
        before_edit = i < first_edit_idx
        
        # ── Empty action ──
        if not action.strip():
            labels.append('!')
            continue
        
        # ── Submit ──
        if fl_lower.startswith('submit'):
            labels.append('S')
            continue
        
        # ── Check for error observation on bash commands ──
        if _is_error_obs(obs) and not fl_lower.startswith('str_replace_editor'):
            labels.append('!')
            continue
        
        # ── str_replace_editor view ──
        if fl_lower.startswith('str_replace_editor view'):
            if _is_directory_target(fl):
                labels.append('O')
            else:
                # File view: is it reviewing own edits?
                parts = fl.split()
                target = parts[2] if len(parts) >= 3 else ''
                if not before_edit and target in edited_files:
                    labels.append('V')  # reviewing own work
                else:
                    labels.append('R')
            continue
        
        # ── str_replace_editor create ──
        if fl_lower.startswith('str_replace_editor create'):
            parts = fl.split()
            target = parts[2] if len(parts) >= 3 else ''
            fname = target.split('/')[-1].lower()
            
            if _is_doc_filename(fname):
                labels.append('V')  # documentation is post-impl activity
            elif _is_repro_filename(fname):
                labels.append('D' if before_edit else 'V')
            elif _is_test_filename(fname):
                labels.append('V')
            elif any(target.startswith(p) for p in ['/app/src/', '/app/lib/', '/app/qutebrowser/']):
                labels.append('I')
            else:
                # Generic create — repro if before edit, verify if after
                labels.append('D' if before_edit else 'V')
            continue
        
        # ── str_replace_editor str_replace / insert ──
        if (fl_lower.startswith('str_replace_editor str_replace') or
            fl_lower.startswith('str_replace_editor insert')):
            parts = fl.split()
            target = parts[2] if len(parts) >= 3 else ''
            fname = target.split('/')[-1].lower()
            
            if _is_repro_filename(fname) or _is_test_filename(fname) or _is_doc_filename(fname):
                labels.append('V')
            else:
                labels.append('I')
            continue
        
        # ── str_replace_editor undo_edit ──
        if fl_lower.startswith('str_replace_editor undo'):
            labels.append('I')
            continue
        
        # ── Everything else is bash or direct commands ──
        cmd = _extract_bash_command(action)
        cmd_lower = cmd.lower().strip()
        
        # applypatch
        if 'applypatch' in cmd_lower:
            labels.append('I')
            continue
        
        # Test suite
        if _is_test_command(cmd):
            labels.append('V')
            continue
        
        # Compile/build check
        if _is_compile_check(cmd):
            labels.append('V')
            continue
        
        # Run a script
        if _is_run_command(cmd):
            target_name = _get_run_target(cmd)
            if _is_repro_filename(target_name):
                labels.append('D' if before_edit else 'V')
            elif _is_test_filename(target_name):
                labels.append('V')
            else:
                # Inline snippet (python -, python -c, node -e) or unknown script
                labels.append('D' if before_edit else 'V')
            continue
        
        # Search/explore commands
        if _is_search_command(cmd):
            labels.append('O')
            continue
        
        # Reading commands (cat, head, tail, sed -n)
        if _is_read_command(cmd):
            labels.append('R')
            continue
        
        # Git commands
        if cmd_lower.strip().startswith('git '):
            labels.append('O' if before_edit else 'V')
            continue
        
        # File management (rm, mv, cp, chmod)
        if any(cmd_lower.strip().startswith(p) for p in ['rm ', 'mv ', 'cp ', 'chmod ']):
            labels.append('V')
            continue
        
        # Install/setup
        if any(kw in cmd_lower for kw in ['pip ', 'npm ', 'go get', 'apt ']):
            labels.append('O')
            continue
        
        # Fallback: before edit = orient, after = verify
        labels.append('O' if before_edit else 'V')
    
    return labels


def classify_file(traj_path):
    """Load a trajectory file and return phase sequence."""
    with open(traj_path) as f:
        data = json.load(f)
    labels = classify_trajectory(data['trajectory'])
    return ''.join(labels), data


if __name__ == '__main__':
    import sys
    import glob
    
    base = '/Users/srihari/work/nilenso/swe-bench-pro-analysis/data'
    
    if len(sys.argv) > 1:
        # Classify a specific file
        seq, data = classify_file(sys.argv[1])
        print(seq)
    else:
        # Demo: classify the ansible task for both models
        inst = 'instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2'
        
        for model in ['gpt5', 'claude45']:
            traj_file = glob.glob(f'{base}/{model}/traj/{inst}/*.traj')[0]
            seq, data = classify_file(traj_file)
            n = len(data['trajectory'])
            print(f"\n{model} ({n} steps): {seq}")
            
            # Show step-by-step
            for i, (step, label) in enumerate(zip(data['trajectory'], seq)):
                fl = step['action'].strip().split('\n')[0][:80]
                print(f"  {label} {i:2d}  {fl}")
