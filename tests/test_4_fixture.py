"""
test for rtanalysis
- in this test, we will create a simulated dataset as a fixture
and use it across multiple tests
- we also create a separate fixture to hold the parameters
"""
import numpy as np
import pytest
from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis


@pytest.fixture
def params():
    return {"meanRT": 2.1, "sdRT": 0.9, "meanAcc": 0.8}


@pytest.fixture
def simulated_data(params):
    return generate_test_df(params["meanRT"], params["sdRT"], params["meanAcc"])


def test_rtanalysis_fit(simulated_data, params):
    rta = RTAnalysis()
    rta.fit(simulated_data.rt, simulated_data.accuracy)
    assert np.allclose(params["meanRT"], rta.meanrt_)
    assert np.allclose(params["meanAcc"], rta.meanacc_)


def test_rtanalysis_checkfail(simulated_data, params):
    rta = RTAnalysis()
    with pytest.raises(ValueError):
        rta.fit(
            simulated_data.rt, simulated_data.accuracy.loc[1:]
        )  # omit first datapoint


# Exercise


def test_rtanalysis_fit_with_cutoff(simulated_data, params):
    rta = RTAnalysis(outlier_cutoff_sd=2)
    rta.fit(simulated_data.rt, simulated_data.accuracy)
    assert np.allclose(params["meanAcc"], rta.meanacc_)
    assert params["meanRT"] > rta.meanrt_


def test_rtanalysis_fit_with_nonseries(simulated_data, params):
    rta = RTAnalysis()
    rta.fit(simulated_data.rt.to_numpy(), simulated_data.accuracy.to_numpy())
    assert np.allclose(params["meanRT"], rta.meanrt_)
    assert np.allclose(params["meanAcc"], rta.meanacc_)
