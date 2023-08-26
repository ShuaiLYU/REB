from .defect_makers import *

import src.DefectMaker.defect_makers as dm
import functools



####################################################################################################################################

def dm_cm1_cp(dataset):
    return functools.partial(dm.get_data_wrapper, dms=dm.get_default_cutpaste_dm(dataset))


def dm_cm1_cps(dataset):
    return functools.partial(dm.get_data_wrapper, dms=dm.get_default_cutpastescar_dm(dataset))


def dm_cm2_cp3way(dataset):
    dms = dm.merge_dict(dm.get_default_cutpaste_dm(dataset), dm.get_default_cutpastescar_dm(dataset))
    return functools.partial(dm.get_data_wrapper, dms=dms)


def get_bezier_cutpaste_dm(dataset):
    dm_name = "bezier_cp"
    shape_name = "BezierRectShapeMaker"
    fill_name = "CutFillMaker"
    assert (hasattr(dataset, "img_size"))
    shape_param = {"img_size": dataset.img_size, "area_ratio_range": [0.02, 0.15], "aspect_ratio": 3.3,
                   "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}
    fill_param = {"dataset": dataset, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT}
    make_param = {"fuse_weight_range": 1, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT, "blur_ksize": 5,
                  "blur_prob": 0.5}
    dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
    return dms


def get_bezierscar_cutpaste_dm(dataset):
    dm_name = "bezier_scar_cp"
    shape_name = "BezierRectScarShapeMaker"
    fill_name = "CutFillMaker"
    assert (hasattr(dataset, "img_size"))
    shape_param = {"width_range": [2 + 5, 16 + 5], "height_range": [10 + 5, 25 + 5], "rotation_range": [-45, 45],
                   "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}
    fill_param = {"dataset": dataset, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT}
    make_param = {"fuse_weight_range": 1, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT, "blur_ksize": 3,
                  "blur_prob": 0.5}
    dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
    return dms


def get_bezierclump_cutpaste_dm(dataset):
    dm_name = "bezier_clump_cp"
    shape_name = "BezierClumpShapeMaker"
    fill_name = "CutFillMaker"
    shape_param = {"img_size": dataset.img_size, "area_ratio_range": [0.02, 0.15], "aspect_ratio": 3,
                   "shape_scale_range": [5, 7]}
    fill_param = {"dataset": dataset, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT}
    make_param = {"fuse_weight_range": (0.5, 1), "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT,
                  "blur_ksize": 3, "blur_prob": 0.5}
    dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
    return dms


def get_bezier_noise_dm(dataset):
    dm_name = "bezier_noise"
    shape_name = "BezierRectShapeMaker"
    fill_name = "RandomNoiseFillMaker"
    assert (hasattr(dataset, "img_size"))
    shape_param = {"img_size": dataset.img_size, "area_ratio_range": [0.02, 0.15], "aspect_ratio": 3.3,
                   "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}
    fill_param = dict(mean_range=[50, 200], mean_step=10, fluct_range=[0, 50], fluect_step=5, scale_range=[0, 3],
                      aspect_ratio_range=[0, 3])
    make_param = {"fuse_weight_range": 1, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT, "blur_ksize": 5,
                  "blur_prob": 0.5}
    dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
    return dms


def get_bezierscar_noise_dm(dataset):
    dm_name = "bezier_scar_noise"
    shape_name = "BezierRectScarShapeMaker"
    fill_name = "RandomNoiseFillMaker"
    assert (hasattr(dataset, "img_size"))
    shape_param = {"width_range": [2 + 5, 16 + 5], "height_range": [10 + 5, 25 + 5], "rotation_range": [-45, 45],
                   "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}
    fill_param = dict(mean_range=[50, 200], mean_step=10, fluct_range=[0, 50], fluect_step=5, scale_range=[0, 3],
                      aspect_ratio_range=[0, 3])
    make_param = {"fuse_weight_range": 1, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT, "blur_ksize": 3,
                  "blur_prob": 0.5}
    dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
    return dms


def get_bezierclump_noise_dm(dataset):
    dm_name = "bezier_clump_noise"
    shape_name = "BezierClumpShapeMaker"
    fill_name = "RandomNoiseFillMaker"
    shape_param = {"img_size": dataset.img_size, "area_ratio_range": [0.02, 0.15], "aspect_ratio": 3,
                   "shape_scale_range": [5, 7]}
    fill_param = dict(mean_range=[50, 200], mean_step=10, fluct_range=[0, 50], fluect_step=5, scale_range=[0, 3],
                      aspect_ratio_range=[0, 3])
    make_param = {"fuse_weight_range": (0.5, 1), "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT,
                  "blur_ksize": 3, "blur_prob": 0.5}
    dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
    return dms


def dm_com2_bezier(dataset):
    dms = {}
    dms = dm.merge_dict(dms, get_bezier_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezier_noise_dm(dataset))
    return functools.partial(dm.get_data_wrapper, dms=dms)


def dm_com2_bezierscar(dataset):
    dms = {}
    dms = dm.merge_dict(dms, get_bezierscar_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierscar_noise_dm(dataset))
    return functools.partial(dm.get_data_wrapper, dms=dms)


def dm_com2_bezierclump(dataset):
    dms = {}
    dms = dm.merge_dict(dms, get_bezierclump_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierclump_noise_dm(dataset))
    return functools.partial(dm.get_data_wrapper, dms=dms)


def dm_com3_noise(dataset):
    dms = {}
    dms = dm.merge_dict(dms, get_bezier_noise_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierscar_noise_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierclump_noise_dm(dataset))

    return functools.partial(dm.get_data_wrapper, dms=dms)


def dm_com3_cp(dataset):
    dms = {}
    dms = dm.merge_dict(dms, get_bezier_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierscar_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierclump_cutpaste_dm(dataset))

    return functools.partial(dm.get_data_wrapper, dms=dms)


def dm_com6(dataset):
    dms = {}
    dms = dm.merge_dict(dms, get_bezier_noise_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierscar_noise_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierclump_noise_dm(dataset))
    dms = dm.merge_dict(dms, get_bezier_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierscar_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezierclump_cutpaste_dm(dataset))

    return functools.partial(dm.get_data_wrapper, dms=dms)


####################################################################################################################################


def dm2_com4(dataset):
    def get_bezier_cutpaste_dm(dataset):
        dm_name = "imp_cp"
        shape_name = "BezierRectShapeMaker"
        fill_name = "CutFillMaker"
        assert (hasattr(dataset, "img_size"))
        shape_param = {"img_size": dataset.img_size, "area_ratio_range": [0.02, 0.15], "aspect_ratio": 3.3,
                       "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}
        fill_param = {"dataset": dataset, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT}
        make_param = {"fuse_weight_range": 1, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT, "blur_ksize": 5,
                      "blur_prob": 0.5}
        dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
        return dms

    def get_bezier_cutpastescar_dm(dataset):
        dm_name = "imp_cps"
        shape_name = "BezierRectScarShapeMaker"
        fill_name = "CutFillMaker"
        assert (hasattr(dataset, "img_size"))
        shape_param = {"width_range": [2 + 5, 16 + 5], "height_range": [10 + 5, 25 + 5], "rotation_range": [-45, 45],
                       "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}
        fill_param = {"dataset": dataset, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT}
        make_param = {"fuse_weight_range": 1, "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT, "blur_ksize": 3,
                      "blur_prob": 0.5}
        dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
        return dms

    def get_clump_noise_dm(dataset):
        dm_name = "clump_noise"
        shape_name = "BezierClumpShapeMaker"
        fill_name = "RandomNoiseFillMaker"
        shape_param = {"img_size": dataset.img_size, "area_ratio_range": [0.02, 0.15], "aspect_ratio": 3,
                       "shape_scale_range": [5, 7]}
        fill_param = dict(mean_range=[50, 200], mean_step=10, fluct_range=[0, 50], fluect_step=5, scale_range=[0, 3],
                          aspect_ratio_range=[0, 3])
        make_param = {"fuse_weight_range": (0.5, 1), "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT,
                      "blur_ksize": 3, "blur_prob": 0.5}
        dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
        return dms

    def get_bezier_NoiseScar_dm(dataset):
        dm_name = "noise_scar"
        shape_name = "BezierRectScarShapeMaker"
        fill_name = "RandomNoiseFillMaker"
        assert (hasattr(dataset, "img_size"))
        shape_param = {"width_range": [2 + 5, 16 + 5], "height_range": [10 + 5, 25 + 5], "rotation_range": [-45, 45],
                       "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}

        fill_param = dict(mean_range=[50, 200], mean_step=10, fluct_range=[0, 50], fluect_step=5, scale_range=[0, 3],
                          aspect_ratio_range=[0, 3])
        make_param = {"fuse_weight_range": (0.5, 1), "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT,
                      "blur_ksize": 3, "blur_prob": 0.5}
        dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
        return dms

    dms = {}
    dms = dm.merge_dict(dms, get_bezier_cutpaste_dm(dataset))
    dms = dm.merge_dict(dms, get_bezier_cutpastescar_dm(dataset))
    dms = dm.merge_dict(dms, get_clump_noise_dm(dataset))
    dms = dm.merge_dict(dms, get_bezier_NoiseScar_dm(dataset))

    return functools.partial(dm.get_data_wrapper, dms=dms)


def dm2_com2_noise(dataset):
    def get_clump_noise_dm(dataset):
        dm_name = "clump_noise"
        shape_name = "BezierClumpShapeMaker"
        fill_name = "RandomNoiseFillMaker"
        shape_param = {"img_size": dataset.img_size, "area_ratio_range": [0.02, 0.15], "aspect_ratio": 3,
                       "shape_scale_range": [5, 7]}
        fill_param = dict(mean_range=[50, 200], mean_step=10, fluct_range=[0, 50], fluect_step=5, scale_range=[0, 3],
                          aspect_ratio_range=[0, 3])
        make_param = {"fuse_weight_range": (0.5, 1), "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT,
                      "blur_ksize": 3, "blur_prob": 0.5}
        dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
        return dms

    def get_bezier_NoiseScar_dm(dataset):
        dm_name = "noise_scar"
        shape_name = "BezierRectScarShapeMaker"
        fill_name = "RandomNoiseFillMaker"
        assert (hasattr(dataset, "img_size"))
        shape_param = {"width_range": [2 + 5, 16 + 5], "height_range": [10 + 5, 25 + 5], "rotation_range": [-45, 45],
                       "bezier_point_num_range": [5, 7], "k_range": [0.2, 0.5]}

        fill_param = dict(mean_range=[50, 200], mean_step=10, fluct_range=[0, 50], fluect_step=5, scale_range=[0, 3],
                          aspect_ratio_range=[0, 3])
        make_param = {"fuse_weight_range": (0.5, 1), "saliency_method": dm.SalienMethod.SALIENCY_CONSTRAINT,
                      "blur_ksize": 3, "blur_prob": 0.5}
        dms = {dm_name: dm.get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
        return dms

    dms = {}
    dms = dm.merge_dict(dms, get_clump_noise_dm(dataset))
    dms = dm.merge_dict(dms, get_bezier_NoiseScar_dm(dataset))

    return functools.partial(dm.get_data_wrapper, dms=dms)



def get_draem_wrapper(dataset):
    def get_draem_dm(dataset):
        dm_name = "draem"
        shape_name = "PerlineShapeMaker"
        fill_name = "CutFillMaker"
        #assert (hasattr(dataset, "img_size"))
        shape_param = {"threshold_set": set([0.5]), "perlin_scale_range": [0, 6]}
        fill_param = {"dataset": dataset, "saliency_method": SalienMethod.NONE}
        make_param = {"fuse_weight_range": (0.2, 1), "saliency_method": SalienMethod.IMAGE_FIT}
        dms = {dm_name: get_defect_maker(dataset, shape_name, fill_name, shape_param, fill_param, make_param)}
        return dms
    return functools.partial(get_data_wrapper, dms=get_draem_dm(dataset))