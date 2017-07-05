#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tempfile
import shutil

import pytest

@pytest.fixture(scope="module")
def temp_dir():
    _temp_dir = tempfile.mkdtemp()
    yield _temp_dir
    shutil.rmtree(_temp_dir)
