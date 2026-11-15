#!/usr/bin/env python
"""DISABLED — raises NotImplementedError.

This script extracted OSV-5M test zips by iterating archive order, which is
NOT a valid query sample (EVALUATION.md 'Data Separation': sampling must be
deterministic and sequence-aware via a manifest that does not exist yet).
Do not re-enable until that manifest is authored; no sampling is implemented
here on purpose.
"""
import sys


def main(*args, **kwargs):
    raise NotImplementedError(
        'prep_queries.py is disabled: sequence-aware query manifests are not '
        'authored yet (see EVALUATION.md). Refusing to sample by archive order.')


if __name__ == '__main__':
    main(*sys.argv[1:])
