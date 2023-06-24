"""A module that contains the result classes.
"""
import abc


class Result(abc.ABC, dict):
    """The abstract base class for results.
    """


class TrainResult(Result):
    """The result of a training phase.
    """


class TestResult(Result):
    """The result of a testing phase.
    """
