import numpy as np

from hac.rgb_witness_policy import policy_features, fit_policy, route


def _fixture(rows=240):
    rng=np.random.default_rng(7)
    labels=np.arange(rows)%2
    anchor=np.tile([.55,.40,.05],(rows,1)).astype(float)
    anchor[labels==1,:2]=[.40,.55]  # initially correct
    candidate=anchor.copy()
    # First60: wrong anchor -> correct candidate (rescues).
    anchor[:60,:2]=anchor[:60,:2][:,::-1]
    # Next60: correct anchor -> wrong candidate (harms). Others do not cross.
    candidate[60:120,:2]=candidate[60:120,:2][:,::-1]
    delta=rng.normal(size=rows)
    geometry=rng.normal(size=(rows,6))
    available=np.ones((rows,2),bool)
    scenarios=np.asarray([f"s{i%6}" for i in range(rows)])
    return labels,anchor,candidate,delta,geometry,available,scenarios


def test_policy_feature_schema_and_support_failure_retains():
    labels,a,c,d,g,av,s=_fixture(20)
    x=policy_features(a,c,d,g,av)
    assert x.shape==(20,19)
    policy,receipt=fit_policy(x,a,c,labels,s)
    assert policy is None and "NO_FIT" in receipt["status"]
    output,choices,_=route(a,c,x,policy)
    assert np.array_equal(output,a) and not choices.any()


def test_policy_fits_or_reports_numerical_issue_on_supported_events():
    labels,a,c,d,g,av,s=_fixture()
    x=policy_features(a,c,d,g,av)
    policy,receipt=fit_policy(x,a,c,labels,s)
    assert receipt["event_rows"] >= 50
    assert policy is not None and receipt["status"]=="RGB_WITNESS_POLICY_FIT_COMPLETE"
    output,choices,_=route(a,c,x,policy)
    assert output.shape==a.shape and choices.shape==(len(a),)
