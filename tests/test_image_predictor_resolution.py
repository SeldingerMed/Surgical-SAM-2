from sam2.sam2_image_predictor import SAM2ImagePredictor


def test_feature_sizes_follow_the_model_resolution():
    model = type("Model", (), {"image_size": 512})()

    predictor = SAM2ImagePredictor(model)

    assert predictor._bb_feat_sizes == [(128, 128), (64, 64), (32, 32)]
