#!/usr/bin/env python

import unittest

# Run all of the tests
loader = unittest.TestLoader()
suite  = loader.discover('tests')
runner = unittest.TextTestRunner()
result = runner.run(suite)

# Can run all the examples by doing the same thing here.
