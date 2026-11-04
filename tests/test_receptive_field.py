from src.lab.receptive_field import RF, chain, upsample, cell_box


def test_two_layer_chain_matches_hand_computation():
    # 3x3 s1 p1 then 3x3 s2 p1: rf = 1 + 2 + 2 = 5, jump = 2, start = 0
    rf = chain([(3, 1, 1), (3, 2, 1)])
    assert rf.size == 5 and rf.jump == 2 and rf.start == 0


def test_one_by_one_conv_and_upsample_leave_size_unchanged():
    rf = chain([(3, 2, 1), (1, 1, 0)])
    assert rf.size == 3 and rf.jump == 2
    up = upsample(rf, 2)
    assert up.size == 3 and up.jump == 1


def test_efficientnet_b0_stem_then_block1_depthwise():
    # stem 3x3 s2 p1 -> rf 3 jump 2; block1 dw 3x3 s2 p1 -> rf 3 + 2*2 = 7, jump 4
    rf = chain([(3, 2, 1), (1, 1, 0), (3, 2, 1)])
    assert rf.size == 7 and rf.jump == 4


def test_cell_box_is_clipped_to_image():
    rf = RF(size=8, jump=4, start=0)
    assert cell_box(rf, 0, 0, 32, 32) == (0, 4, 0, 4)
    assert cell_box(rf, 2, 3, 32, 32) == (4, 12, 8, 16)
