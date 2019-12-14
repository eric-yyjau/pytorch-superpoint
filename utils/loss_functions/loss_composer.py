from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, SpartanDatasetDataType
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss

import torch
from torch.autograd import Variable

def get_loss(pixelwise_contrastive_loss, match_type, 
              image_a_pred, image_b_pred,
              matches_a,     matches_b,
              masked_non_matches_a, masked_non_matches_b,
              background_non_matches_a, background_non_matches_b,
              blind_non_matches_a, blind_non_matches_b):
    """
    This function serves the purpose of:
    - parsing the different types of SpartanDatasetDataType...
    - parsing different types of matches / non matches..
    - into different pixelwise contrastive loss functions

    :return args: loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss
    :rtypes: each pytorch Variables

    """
    if (match_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE).all():
        print "applying SINGLE_OBJECT_WITHIN_SCENE loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE).all():
        print "applying SINGLE_OBJECT_ACROSS_SCENE loss"
        return get_same_object_across_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.DIFFERENT_OBJECT).all():
        print "applying DIFFERENT_OBJECT loss"
        return get_different_object_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            blind_non_matches_a, blind_non_matches_b)


    if (match_type == SpartanDatasetDataType.MULTI_OBJECT).all():
        print "applying MULTI_OBJECT loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT).all():
        print "applying SYNTHETIC_MULTI_OBJECT loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    else:
        raise ValueError("Should only have above scenes?")


def get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                        matches_a,    matches_b,
                                        masked_non_matches_a, masked_non_matches_b,
                                        background_non_matches_a, background_non_matches_b,
                                        blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    pcl = pixelwise_contrastive_loss

    match_loss, masked_non_match_loss, num_masked_hard_negatives =\
        pixelwise_contrastive_loss.get_loss_matched_and_non_matched_with_l2(image_a_pred,         image_b_pred,
                                                                          matches_a,            matches_b,
                                                                          masked_non_matches_a, masked_non_matches_b,
                                                                          M_descriptor=pcl._config["M_masked"])

    if pcl._config["use_l2_pixel_loss_on_background_non_matches"]:
        background_non_match_loss, num_background_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b, 
                background_non_matches_a, background_non_matches_b, M_descriptor=pcl._config["M_background"])    
        
    else:
        background_non_match_loss, num_background_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    background_non_matches_a, background_non_matches_b,
                                                                    M_descriptor=pcl._config["M_background"])
        
        

    blind_non_match_loss = zero_loss()
    num_blind_hard_negatives = 1
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss, num_blind_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=pcl._config["M_masked"])
        


    total_num_hard_negatives = num_masked_hard_negatives + num_background_hard_negatives
    total_num_hard_negatives = max(total_num_hard_negatives, 1)

    if pcl._config["scale_by_hard_negatives"]:
        scale_factor = total_num_hard_negatives

        masked_non_match_loss_scaled = masked_non_match_loss*1.0/max(num_masked_hard_negatives, 1)

        background_non_match_loss_scaled = background_non_match_loss*1.0/max(num_background_hard_negatives, 1)

        blind_non_match_loss_scaled = blind_non_match_loss*1.0/max(num_blind_hard_negatives, 1)
    else:
        # we are not currently using blind non-matches
        num_masked_non_matches = max(len(masked_non_matches_a),1)
        num_background_non_matches = max(len(background_non_matches_a),1)
        num_blind_non_matches = max(len(blind_non_matches_a),1)
        scale_factor = num_masked_non_matches + num_background_non_matches


        masked_non_match_loss_scaled = masked_non_match_loss*1.0/num_masked_non_matches

        background_non_match_loss_scaled = background_non_match_loss*1.0/num_background_non_matches

        blind_non_match_loss_scaled = blind_non_match_loss*1.0/num_blind_non_matches



    non_match_loss = 1.0/scale_factor * (masked_non_match_loss + background_non_match_loss)

    loss = pcl._config["match_loss_weight"] * match_loss + \
    pcl._config["non_match_loss_weight"] * non_match_loss

    

    return loss, match_loss, masked_non_match_loss_scaled, background_non_match_loss_scaled, blind_non_match_loss_scaled

def get_within_scene_loss_triplet(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                        matches_a,    matches_b,
                                        masked_non_matches_a, masked_non_matches_b,
                                        background_non_matches_a, background_non_matches_b,
                                        blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    
    pcl = pixelwise_contrastive_loss

    masked_triplet_loss =\
        pixelwise_contrastive_loss.get_triplet_loss(image_a_pred, image_b_pred, matches_a, 
            matches_b, masked_non_matches_a, masked_non_matches_b, pcl._config["alpha_triplet"])
        
    background_triplet_loss =\
        pixelwise_contrastive_loss.get_triplet_loss(image_a_pred, image_b_pred, matches_a, 
            matches_b, background_non_matches_a, background_non_matches_b, pcl._config["alpha_triplet"])

    total_loss = masked_triplet_loss + background_triplet_loss

    return total_loss, zero_loss(), zero_loss(), zero_loss(), zero_loss()

def get_different_object_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                              blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """

    scale_by_hard_negatives = pixelwise_contrastive_loss.config["scale_by_hard_negatives_DIFFERENT_OBJECT"]
    blind_non_match_loss = zero_loss()
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        M_descriptor = pixelwise_contrastive_loss.config["M_background"]

        blind_non_match_loss, num_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=M_descriptor)
        
        if scale_by_hard_negatives:
            scale_factor = max(num_hard_negatives, 1)
        else:
            scale_factor = max(len(blind_non_matches_a), 1)

        blind_non_match_loss = 1.0/scale_factor * blind_non_match_loss
    loss = blind_non_match_loss
    return loss, zero_loss(), zero_loss(), zero_loss(), blind_non_match_loss

def get_same_object_across_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                              blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    blind_non_match_loss = zero_loss()
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss, num_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=pcl._config["M_masked"], invert=True)

    if pixelwise_contrastive_loss._config["scale_by_hard_negatives"]:
        scale_factor = max(num_hard_negatives, 1)
    else:
        scale_factor = max(len(blind_non_matches_a), 1)

    loss = 1.0/scale_factor * blind_non_match_loss
    blind_non_match_loss_scaled = 1.0/scale_factor * blind_non_match_loss
    return loss, zero_loss(), zero_loss(), zero_loss(), blind_non_match_loss

def zero_loss():
    return Variable(torch.FloatTensor([0]).cuda())

def is_zero_loss(loss):
    return loss.data[0] < 1e-20


