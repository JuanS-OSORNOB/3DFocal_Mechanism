#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Albert Morgan>
import unittest

# Run all of the tests
loader = unittest.TestLoader()
suite  = loader.discover('tests')
runner = unittest.TextTestRunner()
result = runner.run(suite)

# Can run all the examples by doing the same thing here.
