import torch
from torch.autograd import Variable


class PixelwiseContrastiveLoss(object):

    def __init__(self, image_shape, config=None):
        self.type = "pixelwise_contrastive"
        self.image_width  = image_shape[1]
        self.image_height = image_shape[0]

        assert config is not None
        self._config = config

        self._debug_data = dict()

        self._debug = False

    @property
    def debug(self):
        return self._debug

    @property
    def config(self):
        return self._config

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def debug_data(self):
        return self._debug_data

    def get_loss_matched_and_non_matched_with_l2(self, image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b,
                 M_descriptor=None, M_pixel=None, non_match_loss_weight=1.0, use_l2_pixel_loss=None):
        """
        Computes the loss function

        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension


        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2)^2

        loss = match_loss + non_match_loss

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_a
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """

        PCL = PixelwiseContrastiveLoss

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        if M_pixel is None:
            M_pixel = self._config["M_pixel"]


        if use_l2_pixel_loss is None:
            use_l2_pixel_loss = self._config['use_l2_pixel_loss_on_masked_non_matches']


        match_loss, _, _ = PCL.match_loss(image_a_pred, image_b_pred, matches_a, matches_b)



        if use_l2_pixel_loss:
            non_match_loss, num_hard_negatives =\
                self.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b,
                                                       non_matches_a, non_matches_b,
                                                       M_descriptor=M_descriptor,
                                                       M_pixel=M_pixel)
        else:
            # version with no l2 pixel term
            non_match_loss, num_hard_negatives = self.non_match_loss_descriptor_only(image_a_pred, image_b_pred, non_matches_a, non_matches_b, M_descriptor=M_descriptor)



        return match_loss, non_match_loss, num_hard_negatives

    @staticmethod
    def get_triplet_loss(image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b, alpha):
        """
        Computes the loss function

        \sum_{triplets} ||D(I_a, u_a, I_b, u_{b,match})||_2^2 - ||D(I_a, u_a, I_b, u_{b,non-match)||_2^2 + alpha 

        """
        num_matches = matches_a.size()[0]
        num_non_matches = non_matches_a.size()[0]
        multiplier = num_non_matches / num_matches

        ## non_matches_a is already replicated up to be the right size
        ## non_matches_b is also that side
        ## matches_a is just a smaller version of non_matches_a
        ## matches_b is the only thing that needs to be replicated up in size

        matches_b_long =  torch.t(matches_b.repeat(multiplier, 1)).contiguous().view(-1)
                         
        matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        matches_b_descriptors      = torch.index_select(image_b_pred, 1, matches_b_long)
        non_matches_b_descriptors  = torch.index_select(image_b_pred, 1, non_matches_b)

        triplet_losses = (matches_a_descriptors - matches_b_descriptors).pow(2) - (matches_a_descriptors - non_matches_b_descriptors).pow(2) + alpha
        triplet_loss = 1.0 / num_non_matches * torch.clamp(triplet_losses, min=0).sum()

        return triplet_loss

    @staticmethod
    def match_loss(image_a_pred, image_b_pred, matches_a, matches_b, M=1.0, 
                    dist='euclidean', method='1d'): # dist = 'cos'
        """
        Computes the match loss given by

        1/num_matches * \sum_{matches} ||D(I_a, u_a, I_b, u_b)||_2^2

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b

        :return: match_loss, matches_a_descriptors, matches_b_descriptors
        :rtype: torch.Variable(),

        matches_a_descriptors is torch.FloatTensor with shape torch.Shape([num_matches, descriptor_dimension])
        """
        if method == '2d':
            import torch.nn.functional as F
            num_matches = matches_a.size()[0]
            mode = 'bilinear' 
            def sampleDescriptors(image_a_pred, matches_a, mode, norm=False):
                image_a_pred = image_a_pred.unsqueeze(0) # torch [1, D, H, W]
                matches_a.unsqueeze_(0).unsqueeze_(2)
                matches_a_descriptors = F.grid_sample(image_a_pred, matches_a, mode=mode, align_corners=True)
                matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0,1)
                
                # print("image_a_pred: ", image_a_pred.shape)
                # print("matches_a: ", matches_a.shape)
                # print("matches_a: ", matches_a)
                # print("matches_a_descriptors: ", matches_a_descriptors)
                if norm:
                    dn = torch.norm(matches_a_descriptors, p=2, dim=1) # Compute the norm of b_descriptors
                    matches_a_descriptors = matches_a_descriptors.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
                return matches_a_descriptors

            # image_b_pred = image_b_pred.unsqueeze(0) # torch [1, D, H, W]
            # matches_b.unsqueeze_(0).unsqueeze_(2)
            # matches_b_descriptors = F.grid_sample(image_b_pred, matches_b, mode=mode)
            # matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0,1)
            norm = False
            matches_a_descriptors = sampleDescriptors(image_a_pred, matches_a, mode, norm=norm)
            matches_b_descriptors = sampleDescriptors(image_b_pred, matches_b, mode, norm=norm)
        else:
            num_matches = matches_a.size()[0]
            matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
            matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up 
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(matches_a) == 1:
            matches_a_descriptors = matches_a_descriptors.unsqueeze(0)
            matches_b_descriptors = matches_b_descriptors.unsqueeze(0)

        if dist == 'cos':
            # print("dot product: ", (matches_a_descriptors * matches_b_descriptors).shape)
            match_loss = torch.clamp(M - (matches_a_descriptors * matches_b_descriptors).sum(dim=-1), min=0)
            match_loss = 1.0 / num_matches * match_loss.sum()
        else:
            match_loss = 1.0 / num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

        return match_loss, matches_a_descriptors, matches_b_descriptors


    @staticmethod
    def non_match_descriptor_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, M=0.5, invert=False, dist='euclidear'):
        """
        Computes the max(0, M - D(I_a,I_b,u_a,u_b))^2 term

        This is effectively:       "a and b should be AT LEAST M away from each other"
        With invert=True, this is: "a and b should be AT MOST  M away from each other" 

         :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :param M: the margin
        :type M: float
        :return: torch.FloatTensor with shape torch.Shape([num_non_matches])
        :rtype:
        """

        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a).squeeze()
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b).squeeze()

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up 
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(non_matches_a) == 1:
            non_matches_a_descriptors = non_matches_a_descriptors.unsqueeze(0)
            non_matches_b_descriptors = non_matches_b_descriptors.unsqueeze(0)

        norm_degree = 2
        if dist == 'cos':
            non_match_loss = (non_matches_a_descriptors * non_matches_b_descriptors).sum(dim=-1)
        else:
            non_match_loss = (non_matches_a_descriptors - non_matches_b_descriptors).norm(norm_degree, 1)
        if not invert:
            non_match_loss = torch.clamp(M - non_match_loss, min=0).pow(2)
        else:
            if dist == 'cos':
                non_match_loss = torch.clamp(non_match_loss - M, min=0)
            else:
                non_match_loss = torch.clamp(non_match_loss - M, min=0).pow(2)

        hard_negative_idxs = torch.nonzero(non_match_loss)
        num_hard_negatives = len(hard_negative_idxs)

        return non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors

    def non_match_loss_with_l2_pixel_norm(self, image_a_pred, image_b_pred, matches_b,
                                          non_matches_a, non_matches_b, M_descriptor=0.5,
                                          M_pixel=None):

        """

        Computes the total non_match_loss with an l2_pixel norm term

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_a
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a

        :param M_descriptor: margin for descriptor loss term
        :type M_descriptor: float
        :param M_pixel: margin for pixel loss term
        :type M_pixel: float
        :return: non_match_loss, num_hard_negatives
        :rtype: torch.Variable, int
        """

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        if M_pixel is None:
            M_pixel = self._config["M_pixel"]

        PCL = PixelwiseContrastiveLoss

        num_non_matches = non_matches_a.size()[0]

        non_match_descriptor_loss, num_hard_negatives, _, _ = PCL.non_match_descriptor_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, M=M_descriptor)

        non_match_pixel_l2_loss, _, _ = self.l2_pixel_loss(matches_b, non_matches_b, M_pixel=M_pixel)




        non_match_loss = (non_match_descriptor_loss * non_match_pixel_l2_loss).sum()

        if self.debug:
            self._debug_data['num_hard_negatives'] = num_hard_negatives
            self._debug_data['fraction_hard_negatives'] = num_hard_negatives * 1.0/num_non_matches


        return non_match_loss, num_hard_negatives

    def non_match_loss_descriptor_only(self, image_a_pred, image_b_pred, non_matches_a, non_matches_b, M_descriptor=0.5, invert=False):
        """
        Computes the non-match loss, only using the desciptor norm
        :param image_a_pred:
        :type image_a_pred:
        :param image_b_pred:
        :type image_b_pred:
        :param non_matches_a:
        :type non_matches_a:
        :param non_matches_b:
        :type non_matches_b:
        :param M:
        :type M:
        :return: non_match_loss, num_hard_negatives
        :rtype: torch.Variable, int
        """
        PCL = PixelwiseContrastiveLoss

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        non_match_loss_vec, num_hard_negatives, _, _ = PCL.non_match_descriptor_loss(image_a_pred, image_b_pred, non_matches_a,
                                                                 non_matches_b, M=M_descriptor, invert=invert)

        num_non_matches = long(non_match_loss_vec.size()[0])


        non_match_loss = non_match_loss_vec.sum()

        if self._debug:
            self._debug_data['num_hard_negatives'] = num_hard_negatives
            self._debug_data['fraction_hard_negatives'] = num_hard_negatives * 1.0/num_non_matches

        return non_match_loss, num_hard_negatives


    def l2_pixel_loss(self, matches_b, non_matches_b, M_pixel=None):
        """
        Apply l2 loss in pixel space.

        This weights non-matches more if they are "far away" in pixel space.

        :param matches_b: A torch.LongTensor with shape torch.Shape([num_matches])
        :param non_matches_b: A torch.LongTensor with shape torch.Shape([num_non_matches])
        :return l2 loss per sample: A torch.FloatTensorof with shape torch.Shape([num_matches])
        """

        if M_pixel is None:
            M_pixel = self._config['M_pixel']

        num_non_matches_per_match = len(non_matches_b)/len(matches_b)

        ground_truth_pixels_for_non_matches_b = torch.t(matches_b.repeat(num_non_matches_per_match,1)).contiguous().view(-1,1)

        ground_truth_u_v_b = self.flattened_pixel_locations_to_u_v(ground_truth_pixels_for_non_matches_b)
        sampled_u_v_b      = self.flattened_pixel_locations_to_u_v(non_matches_b.unsqueeze(1))

        # each element is always within [0,1], you have 1 if you are at least M_pixel away in
        # L2 norm in pixel space
        norm_degree = 2
        squared_l2_pixel_loss = 1.0/M_pixel * torch.clamp((ground_truth_u_v_b - sampled_u_v_b).float().norm(norm_degree,1), max=M_pixel)


        return squared_l2_pixel_loss, ground_truth_u_v_b, sampled_u_v_b
        

    
    def flattened_pixel_locations_to_u_v(self, flat_pixel_locations):
        """
        :param flat_pixel_locations: A torch.LongTensor of shape torch.Shape([n,1]) where each element
         is a flattened pixel index, i.e. some integer between 0 and 307,200 for a 640x480 image

        :type flat_pixel_locations: torch.LongTensor

        :return A torch.LongTensor of shape (n,2) where the first column is the u coordinates of
        the pixel and the second column is the v coordinate

        """
        u_v_pixel_locations = flat_pixel_locations.repeat(1,2)
        u_v_pixel_locations[:,0] = u_v_pixel_locations[:,0]%self.image_width 
        u_v_pixel_locations[:,1] = u_v_pixel_locations[:,1]/self.image_width
        return u_v_pixel_locations

    def get_l2_pixel_loss_original(self):
        pass

    def get_loss_original(self, image_a_pred, image_b_pred, matches_a,
                          matches_b, non_matches_a, non_matches_b,
                          M_margin=0.5, non_match_loss_weight=1.0):

        # this is pegged to it's implemenation at sha 87abdb63bb5b99d9632f5c4360b5f6f1cf54245f
        """
        Computes the loss function
        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension
        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2^2 )
        loss = match_loss + non_match_loss
        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """

        num_matches = matches_a.size()[0]
        num_non_matches = non_matches_a.size()[0]


        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)

        match_loss = 1.0/num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

        # add loss via non_matches
        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
        pixel_wise_loss = (non_matches_a_descriptors - non_matches_b_descriptors).pow(2).sum(dim=2)
        pixel_wise_loss = torch.add(torch.neg(pixel_wise_loss), M_margin)
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        non_match_loss = non_match_loss_weight * 1.0/num_non_matches * torch.max(zeros_vec, pixel_wise_loss).sum()

        loss = match_loss + non_match_loss

        return loss, match_loss, non_match_loss
