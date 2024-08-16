import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips as lpips_module
import matplotlib.pyplot as plt

from fitvid.model.nvae import ModularEncoder, ModularDecoder, GraspedFCN
from fitvid.model.depth_predictor import DepthPredictor
from fitvid.utils.depth_utils import pixel_wise_loss, pixel_wise_loss_segmented
from fitvid.utils.pytorch_metrics import (
    psnr,
    lpips,
    ssim,
    tv,
    fvd,
)

import piq


class MultiGaussianLSTM(nn.Module):
    """Multi layer lstm with Gaussian output."""

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(MultiGaussianLSTM, self).__init__()

        # assert num_layers == 1
        # print("input sizeeeee, hidden sizeeee: ", input_size, hidden_size)
        self.embed = nn.Linear(input_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_size)
        self.logvar = nn.Linear(hidden_size, output_size)
        self.layers_0 = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers
        )
        # init hidden state is init to zero

    def forward(self, x, states):
        # assume x to only contain one timestep i.e. (bs, feature_dim)
        # added by Arpit
        x = x.float()

        x = self.embed(x)
        x = x.view((1,) + x.shape)
        x, new_states = self.layers_0(x, states)
        mean = self.mean(x)[0]
        logvar = self.logvar(x)[0]

        epsilon = torch.normal(mean=0, std=1, size=mean.shape).cuda()
        var = torch.exp(0.5 * logvar)
        z_t = mean + var * epsilon
        return (z_t, mean, logvar), new_states


def init_weights_lecun(m):
    """
    Perform LeCun normal initialization for parameters of module m
    Since the default Flax initialization uses LeCun uniform, unlike pytorch default, we use this to try to match the
    official implementation.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialize weights to LeCun normal initialization
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = np.sqrt(1 / fan_in)
        nn.init.trunc_normal_(m.weight, mean=0, std=std, a=-2 * std, b=2 * std)
        # Initialize biases to zero
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        # Handle LSTMs. In jax, the kernels which transform the input are initialized with LeCun normal, and the
        # ones which transform the hidden state are initialized with orthogonal.
        for name, param in m.named_parameters():
            if "weight_hh" in name:
                for i in range(0, param.shape[0], param.shape[0] // 4):
                    nn.init.orthogonal_(param[i : i + param.shape[0] // 4])
            elif "weight_ih" in name:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
                std = np.sqrt(1 / fan_in)
                nn.init.trunc_normal_(param, mean=0, std=std, a=-2 * std, b=2 * std)
            elif "bias" in name:
                nn.init.zeros_(param)

class GraspedModel(nn.Module):

    def __init__(self, **kwargs):
        super(GraspedModel, self).__init__()
        # print("kwargs: ", kwargs)
        self.config = kwargs
        model_kwargs = kwargs["model_kwargs"]
        self.n_past = model_kwargs["n_past"]
        self.action_conditioned = model_kwargs["action_conditioned"]
        self.beta = kwargs["beta"]
        self.stochastic = model_kwargs["stochastic"]
        self.is_inference = kwargs["is_inference"]
        self.loss_weights = kwargs["loss_weights"]
        self.loss_weights["kld"] = self.beta
        self.multistep = kwargs["multistep"]
        self.z_dim = model_kwargs["z_dim"]
        self.num_video_channels = model_kwargs.get("video_channels", 3)
        self.model_fitvid = None

        if self.stochastic:
            self.prior = MultiGaussianLSTM(
                input_size=model_kwargs["g_dim"],
                output_size=model_kwargs["z_dim"],
                hidden_size=model_kwargs["rnn_size"],
                num_layers=1,
            )
            self.posterior = MultiGaussianLSTM(
                input_size=model_kwargs["g_dim"],
                output_size=model_kwargs["z_dim"],
                hidden_size=model_kwargs["rnn_size"],
                num_layers=1,
            )
        else:
            self.prior, self.posterior = None, None

        input_size = self.get_input_size(
            model_kwargs["g_dim"], model_kwargs["action_size"], model_kwargs["z_dim"], model_kwargs["grasped_dim"]
        )
        self.frame_predictor = MultiGaussianLSTM(
            input_size=input_size,
            output_size=model_kwargs["g_dim"],
            hidden_size=model_kwargs["rnn_size"],
            num_layers=2,
        )

        self.grasped_fcn = GraspedFCN()

        if model_kwargs.get("lecun_initialization", None):
            self.apply(init_weights_lecun)

    def set_video_prediction_model(self, model_fitvid):
        self.model_fitvid = model_fitvid
        for param in self.model_fitvid.parameters():
            param.requires_grad = False
    
    def get_input(self, hidden, action, z, grasped=None):
        # print("hidden, action, z, grasped:", hidden.shape, action.shape, z.shape, grasped.shape)
        inp = [hidden]
        if self.action_conditioned:
            inp += [action]
        if self.stochastic:
            inp += [z]
        if grasped is not None:
            inp += [grasped]
        return torch.cat(inp, dim=1)

    def get_input_size(self, hidden_size, action_size, z_size, grasped_size=None):
        inp = hidden_size
        if self.action_conditioned:
            inp += action_size
        if self.stochastic:
            inp += z_size
        if grasped_size is not None:
            inp += grasped_size
        return inp
    
    def predict_next_state(self, video, actions, grasped, hidden, skips):
        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state_grasped = post_state_grasped = prior_state_grasped = None
        pred_state_rgb = post_state_rgb = prior_state_rgb = None

        kld, means, logvars = torch.tensor(0).to(video), [], []
        # training
        h_preds = []
        if self.training and self.multistep:
            rgb_preds = []
            grasped_preds = []
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                
                if i > self.n_past:
                    # convert the predicted image to segmentation image again
                    rgb_pred = rgb_pred.permute(0,2,3,1)
                    logSoftmax = torch.nn.LogSoftmax(dim=-1)
                    out = logSoftmax(rgb_pred)
                    inds = torch.argmax(out, axis=-1)
                    rgb_pred_seg_img = torch.nn.functional.one_hot(inds, num_classes=rgb_pred.shape[-1]).type(torch.cuda.FloatTensor)
                    rgb_pred = rgb_pred_seg_img.permute(0,3,1,2)

                    h, _ = self.model_fitvid.encoder(rgb_pred)
                
                if self.stochastic:
                    (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                    (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                else:
                    z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)                # print("shapes, h, actions, z_t: ", h.shape, actions.shape, z_t.shape)
                
                inp_grasped = self.get_input(h, actions[:, i - 1], z_t, grasped[:, i - 1]) # TODO: Maybe Change the grasped value here
                # print("inp_grasped.shape: ", inp_grasped.shape)
                (_, h_pred_grasped, _), pred_state_grasped = self.frame_predictor(inp_grasped, pred_state_grasped)
                h_pred_grasped = torch.sigmoid(h_pred_grasped)  
                grasped_pred = self.grasped_fcn(h_pred_grasped)
                # h_preds.append(h_pred)
                # if self.stochastic:
                #     means.append(mu)
                #     logvars.append(logvar)
                #     kld += self.kl_divergence(
                #         mu, logvar, prior_mu, prior_logvar, batch_size
                #     )
                
                # decode the rgb prediction from the trained Frame Preidctor (LSTM) + Decoder from model_fitvid
                inp_rgb = self.get_input(h, actions[:, i - 1], z_t)
                # print("inp_rgb.shape: ", inp_rgb.shape)
                (_, h_pred_rgb, _), pred_state_rgb = self.model_fitvid.frame_predictor(inp_rgb, pred_state_rgb)
                with torch.no_grad():
                    h_pred_rgb = torch.sigmoid(h_pred_rgb)
                rgb_pred = self.model_fitvid.decoder(h_pred_rgb, skips, has_time_dim=False)
                
                # print("pred: ", pred.shape)
                rgb_preds.append(rgb_pred)
                grasped_preds.append(grasped_pred)
            rgb_preds = torch.stack(rgb_preds, axis=1)
            grasped_preds = torch.stack(grasped_preds, axis=1)
        else:
            pass
            # for i in range(1, video_len):
            #     h, h_target = hidden[:, i - 1], hidden[:, i]
            #     if self.stochastic:
            #         (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
            #         (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
            #     else:
            #         z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)                
            #     inp = self.get_input(h, actions[:, i - 1], z_t, grasped[:, i - 1])
            #     (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
            #     h_pred = torch.sigmoid(h_pred)  # TODO notice
            #     h_preds.append(h_pred)
            #     if self.stochastic:
            #         means.append(mu)
            #         logvars.append(logvar)
            #         kld += self.kl_divergence(
            #             mu, logvar, prior_mu, prior_logvar, batch_size
            #         )
            # h_preds = torch.stack(h_preds, axis=1)
            # preds = self.decoder(h_preds, skips)
            # grasped_preds = self.grasped_fcn(h_preds)

        if self.stochastic:
            means = torch.stack(means, axis=1)
            logvars = torch.stack(logvars, axis=1)
        else:
            means, logvars = torch.zeros(h.shape[0], video_len - 1, 1).to(
                h
            ), torch.zeros(h.shape[0], video_len - 1, 1).to(h)
        return rgb_preds, kld, means, logvars, grasped_preds
    
    def forward(
        self,
        video,
        actions,
        grasped,
        compute_metrics=False,
    ):
        batch_size, video_len = video.shape[0], video.shape[1]
        # print("11video.shape: ", video.shape)
        video = video.view(
            (batch_size * video_len,) + video.shape[2:]
        )  # collapse first two dims
        # print("22video.shape: ", video.shape)
        hidden, skips = self.model_fitvid.encoder(video)
        # print("11hidden, skips: ", hidden.shape)
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        # print("22hidden, skips: ", hidden.shape)
        video = video.view(
            (
                batch_size,
                video_len,
            )
            + video.shape[1:]
        )  # reconstruct first two dims
        # print("33video.shape: ", video.shape)

        grasped_gt = grasped

        skips = {
            k: skips[k].view(
                (
                    batch_size,
                    video_len,
                )
                + tuple(skips[k].shape[1:])
            )[:, self.n_past - 1]
            for k in skips.keys()
        }
        preds, kld, means, logvars, grasped_preds = self.predict_next_state(video, actions, grasped, hidden, skips)
        # print("grasped_preds, grasped_gt: ", grasped_preds.shape, grasped_gt.shape)
        loss, preds, metrics = self.compute_loss(
            preds,
            video,
            kld,
            grasped_preds,
            grasped_gt,
            compute_metrics=compute_metrics,
        )
        metrics.update(
            {
                "hist/mean": means,
                "hist/logvars": logvars,
            }
        )
        return loss, preds, metrics

    def compute_loss(
        self,
        preds,
        video,
        kld,
        grasped_preds,
        grasped_gt,
        compute_metrics=False,
    ):
        total_loss = 0
        metrics = dict()
        preds = dict(rgb=preds)
        for loss, weight in self.loss_weights.items():
            # print("loss, weight: ", loss, weight)
            # added by Arpit
            if loss == 'grasped':
                grasped_loss = nn.BCEWithLogitsLoss()(grasped_preds, grasped_gt[:, 1:])
                total_loss += weight * grasped_loss
                metrics["loss/grasped"] = grasped_loss
            elif loss == "rgb":
                    # initialize mask to be a torch tensor of all ones with same shape as video
                    with torch.no_grad():
                        mse_per_sample = pixel_wise_loss(
                            preds["rgb"],
                            video[:, 1:],
                            loss="l2",
                            reduce_batch=False,
                            mask=None,
                        )
                    metrics["loss/mse"] = mse_per_sample.mean().detach()
                    metrics["loss/mse_per_sample"] = mse_per_sample.detach()

        # Metrics
        metrics.update(
            {
                "loss/all": total_loss,
            }
        )
        # if compute_metrics:
        #     if segmentation is not None:
        #         metrics.update(
        #             self.compute_metrics(
        #                 preds["rgb"], video[:, 1:], segmentation[:, 1:]
        #             )
        #         )
        #     else:
        #         metrics.update(self.compute_metrics(preds["rgb"], video[:, 1:]))

        return total_loss, preds, metrics
    
    def evaluate(self, batch, compute_metrics=False):
        ag_metrics, ag_preds, ag_grasped_preds = self._evaluate(
            batch, compute_metrics, autoregressive=True
        )
        # non_ag_metrics, non_ag_preds, non_ag_grasped_preds = self._evaluate(
        #     batch, compute_metrics, autoregressive=False
        # )
        ag_metrics = {f"ag/{k}": v for k, v in ag_metrics.items()}
        # non_ag_metrics = {f"non_ag/{k}": v for k, v in non_ag_metrics.items()}
        metrics = {**ag_metrics} #  **non_ag_metrics
        return metrics, dict(ag=ag_preds), dict(ag=ag_grasped_preds) #non_ag=non_ag_preds ; non_ag=non_ag_grasped_preds

    def _evaluate(self, batch, compute_metrics=False, autoregressive=True):
        """Predict the full video conditioned on the first self.n_past frames."""
        video, actions, segmentation, grasped = (
            batch["video"],
            batch["actions"],
            batch.get("segmentation", None),
            batch["grasped"]
        )
        # print("GT grasped: ", grasped)
        # print("in evaluate: video, grasped, actions", video.shape, grasped.shape, actions.shape)
        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state_grasped = prior_state_grasped = post_state_grasped = None
        pred_state_rgb = prior_state_rgb = post_state_rgb = None
        video = video.view(
            (batch_size * video_len,) + video.shape[2:]
        )  # collapse first two dims
        # print("video.shape: ", video.shape)
        hidden, skips = self.model_fitvid.encoder(video)
        # print("hidden, skips: ", hidden.shape, skips.keys())
        # print("self.n_past: ", self.n_past)
        skips = {
            k: skips[k].view(
                (
                    batch_size,
                    video_len,
                )
                + tuple(skips[k].shape[1:])
            )[:, self.n_past - 1]
            for k in skips.keys()
        }
        # print("skpis: ", skips.keys())
        # evaluating
        rgb_preds = []
        grasped_preds = []
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        # print("self.n_past:", self.n_past)
        if autoregressive:
            for i in range(1, video_len):
                h, _ = hidden[:, i - 1], hidden[:, i]
                grasped_state = grasped[:, i - 1]
                
                if i > self.n_past:
                    # print("using previous predicted value", i, self.n_past)
                    # input()

                    # convert the predicted image to segmentation image again
                    rgb_pred = rgb_pred.permute(0,2,3,1)
                    logSoftmax = torch.nn.LogSoftmax(dim=-1)
                    out = logSoftmax(rgb_pred)
                    inds = torch.argmax(out, axis=-1)
                    rgb_pred_seg_img = torch.nn.functional.one_hot(inds, num_classes=rgb_pred.shape[-1]).type(torch.cuda.FloatTensor)
                    rgb_pred = rgb_pred_seg_img.permute(0,3,1,2)

                    h, _ = self.model_fitvid.encoder(rgb_pred)
                    grasped_state = torch.round(grasped_pred)
                
                if self.stochastic:
                    (z_t, prior_mu, prior_logvar), prior_state = self.prior(
                        h, prior_state
                    )
                else:
                    z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)

                # print("grasped_state.shape: ", grasped_state.shape)
                inp_grasped = self.get_input(h, actions[:, i - 1], z_t, grasped_state)
                (_, h_pred_grasped, _), pred_state_grasped = self.frame_predictor(inp_grasped, pred_state_grasped)
                h_pred_grasped = torch.sigmoid(h_pred_grasped) 
                grasped_pred = self.grasped_fcn(h_pred_grasped)
                grasped_pred = torch.sigmoid(grasped_pred) 

                # decode the rgb prediction from the trained Frame Preidctor (LSTM) + Decoder from model_fitvid
                inp_rgb = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred_rgb, _), pred_state_rgb = self.model_fitvid.frame_predictor(inp_rgb, pred_state_rgb)
                with torch.no_grad():
                    h_pred_rgb = torch.sigmoid(h_pred_rgb)
                rgb_pred = self.model_fitvid.decoder(h_pred_rgb[None, :], skips)[0]
                
                rgb_preds.append(rgb_pred)
                grasped_preds.append(grasped_pred)
            rgb_preds = torch.stack(rgb_preds, axis=1)
            grasped_preds = torch.stack(grasped_preds, axis=1)
        else:
            h_preds = []
            kld = torch.tensor(0).to(video)
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                if self.stochastic:
                    (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                    (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                else:
                    z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)                
                inp = self.get_input(h, actions[:, i - 1], z_t, grasped[:, i - 1])
                # print("2inp.shapeeeeee: ", inp.shape)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
                if self.stochastic:
                    kld += self.kl_divergence(
                        mu, logvar, prior_mu, prior_logvar, batch_size
                    )
            h_preds = torch.stack(h_preds, axis=1)
            preds = self.decoder(h_preds, skips)
            grasped_preds = self.grasped_fcn(h_preds)
            grasped_preds = torch.sigmoid(grasped_preds) 


        video = video.view(
            (
                batch_size,
                video_len,
            )
            + video.shape[1:]
        )  # reconstuct first two dims
        mse_per_sample = pixel_wise_loss(
            rgb_preds, video[:, 1:], loss="l2", reduce_batch=False
        )
        mse = mse_per_sample.mean()
        l1_loss_per_sample = pixel_wise_loss(
            rgb_preds, video[:, 1:], loss="l1", reduce_batch=False
        )
        l1_loss = l1_loss_per_sample.mean()

        # add grasped state loss
        grasped_preds = grasped_preds.to(torch.float64)
        temp_grasped_preds = grasped_preds.detach().clone()
        temp_grasped_preds = torch.round(temp_grasped_preds)
        # print("grasped_preds: ", torch.squeeze(grasped_preds)[0], grasped_preds.dtype)
        # print("temp_grasped_preds: ", torch.squeeze(temp_grasped_preds)[0], temp_grasped_preds[0,0,0].dtype)
        # print("gt grasped: ", torch.squeeze(grasped[0, 1:]), grasped[0,0,0].dtype)
        output_error = torch.sum(grasped[:, 1:] != temp_grasped_preds)
        with torch.no_grad():
            bce_error = nn.BCELoss()(grasped_preds, grasped[:, 1:])
        # print("output_error, bce_error: ", output_error, bce_error)
        # for b in range(grasped.shape[0]):
        #     if torch.sum(grasped[b, 1:] != temp_grasped_preds[b]) > 0:
        #         print("GT, pred: ", torch.squeeze(grasped[b, 1:]), torch.squeeze(temp_grasped_preds[b]))

        # input()

        metrics = {
            "loss/mse": mse,
            "loss/mse_per_sample": mse_per_sample,
            "loss/l1_loss": l1_loss,
            "loss/l1_loss_per_sample": l1_loss_per_sample,
            "loss/bce": bce_error.to(torch.float64),
            "loss/mismatches": output_error.to(torch.float64)
        }

        # if compute_metrics:
        #     if segmentation is not None:
        #         metrics.update(
        #             self.compute_metrics(preds, video[:, 1:], segmentation[:, 1:])
        #         )
        #     else:
        #         metrics.update(self.compute_metrics(preds, video[:, 1:]))

        rgb_preds = dict(rgb=rgb_preds)

        return metrics, rgb_preds, grasped_preds

    def test(self, batch):
        """Predict the full video conditioned on the first self.n_past frames."""
        video, actions, grasped = batch["video"], batch["actions"], batch["grasped"]
        print("self.n_past: ", self.n_past)
        print("video, actions, grasped: ", video.shape, actions.shape, grasped.shape)
        
        # # for visualizing the input image (debugging)
        # temp = video.permute(0, 1, 3, 4, 2)
        # temp = torch.argmax(temp, axis=-1).cpu()
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2,2)
        # ax[0][0].imshow(temp[0,0])
        # ax[0][1].imshow(temp[13,0])
        # ax[1][0].imshow(temp[27,0])
        # ax[1][1].imshow(temp[146,0])
        # plt.show()

        batch_size, video_len = video.shape[0], video.shape[1]
        action_len = actions.shape[1]
        pred_state_grasped = prior_state_grasped = None
        pred_state_rgb = prior_state_rgb = None
        video = video.view(
            (batch_size * video_len,) + video.shape[2:]
        )  # collapse first two dims
        hidden, skips = self.model_fitvid.encoder(video)
        skips = {
            k: skips[k].view(
                (
                    batch_size,
                    video_len,
                )
                + tuple(skips[k].shape[1:])
            )[:, self.n_past - 1]
            for k in skips.keys()
        }
        # evaluating
        rgb_preds = []
        grasped_preds = []
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        for i in range(1, action_len + 1):
            if i <= self.n_past:
                h = hidden[:, i - 1]
                grasped_state = grasped[:, i - 1]
                # for visualizing the next segmentation image prediction 
                # init_image = torch.argmax(video.cpu().permute(0,2,3,1), axis=-1)
            if i > self.n_past:
                
                # convert the predicted image to segmentation image again
                rgb_pred = rgb_pred.permute(0,2,3,1)
                logSoftmax = torch.nn.LogSoftmax(dim=-1)
                out = logSoftmax(rgb_pred)
                inds = torch.argmax(out, axis=-1)
                rgb_pred_seg_img = torch.nn.functional.one_hot(inds, num_classes=rgb_pred.shape[-1]).type(torch.cuda.FloatTensor)
                rgb_pred = rgb_pred_seg_img.permute(0,3,1,2)

                # visualizing the next segmentation image prediction
                # print("action: ", actions[199])
                # pred_image = torch.argmax(rgb_pred.cpu().permute(0,2,3,1), axis=-1)
                # fig, ax = plt.subplots(3,3)
                # ax[0][0].imshow(init_image[199])
                # ax[0][1].imshow(pred_image[199])
                # ax[0][2].imshow(pred_image[3])
                # ax[1][0].imshow(pred_image[53])
                # ax[1][1].imshow(pred_image[87])
                # ax[1][2].imshow(pred_image[25])
                # ax[2][0].imshow(pred_image[112])
                # ax[2][1].imshow(pred_image[145])
                # ax[2][2].imshow(pred_image[167])
                # plt.show()
                # fig, ax = plt.subplots(1,2)
                # ax[0].imshow(init_image[199])
                # ax[1].imshow(pred_image[199])
                # plt.show()
                # init_image = pred_image.clone()

                h, _ = self.model_fitvid.encoder(rgb_pred)
                grasped_state = torch.round(grasped_pred)

            if self.stochastic:
                (z_t, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
            else:
                z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)
            
            # inp = self.get_input(h, actions[:, i - 1], z_t, g)
            # (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
            # h_pred = torch.sigmoid(h_pred)  # TODO notice
            # pred = self.decoder(h_pred[None, :], skips)[0]
            # grasped_pred = self.grasped_fcn(h_pred)
            # grasped_pred = torch.sigmoid(grasped_pred) 

            inp_grasped = self.get_input(h, actions[:, i - 1], z_t, grasped_state)
            (_, h_pred_grasped, _), pred_state_grasped = self.frame_predictor(inp_grasped, pred_state_grasped)
            h_pred_grasped = torch.sigmoid(h_pred_grasped) 
            grasped_pred = self.grasped_fcn(h_pred_grasped)
            grasped_pred = torch.sigmoid(grasped_pred) 

            # decode the rgb prediction from the trained Frame Preidctor (LSTM) + Decoder from model_fitvid
            inp_rgb = self.get_input(h, actions[:, i - 1], z_t)
            (_, h_pred_rgb, _), pred_state_rgb = self.model_fitvid.frame_predictor(inp_rgb, pred_state_rgb)
            with torch.no_grad():
                h_pred_rgb = torch.sigmoid(h_pred_rgb)
            rgb_pred = self.model_fitvid.decoder(h_pred_rgb[None, :], skips)[0]

            rgb_preds.append(rgb_pred)
            grasped_preds.append(grasped_pred)
        
        rgb_preds = torch.stack(rgb_preds, axis=1)
        grasped_preds = torch.stack(grasped_preds, axis=1)
        
        video = video.view(
            (
                batch_size,
                video_len,
            )
            + video.shape[1:]
        )  # reconstuct first two dims
        return rgb_preds, grasped_preds 
    
    def load_parameters(self, path):
        # load everything
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint {path}")
    
# --------------------------------------------------------------------------------------------------

class FitVid(nn.Module):
    """FitVid video predictor."""

    def __init__(self, **kwargs):
        # print("in fitvid init!!")
        # print("kwargs: ", kwargs)
        super(FitVid, self).__init__()
        self.config = kwargs
        model_kwargs = kwargs["model_kwargs"]
        self.n_past = model_kwargs["n_past"]
        self.action_conditioned = model_kwargs["action_conditioned"]
        self.beta = kwargs["beta"]
        self.stochastic = model_kwargs["stochastic"]
        self.is_inference = kwargs["is_inference"]
        self.skip_type = model_kwargs["skip_type"]
        self.loss_weights = kwargs["loss_weights"]
        self.loss_weights["kld"] = self.beta
        self.multistep = kwargs["multistep"]
        self.z_dim = model_kwargs["z_dim"]
        self.num_video_channels = model_kwargs.get("video_channels", 3)
        self.use_grasped = kwargs["use_grasped"]

        first_block_shape = [model_kwargs["first_block_shape"][-1]] + model_kwargs[
            "first_block_shape"
        ][:2]
        print("num_input_channels: ", self.num_video_channels)
        # input()
        self.encoder = ModularEncoder(
            stage_sizes=model_kwargs["stage_sizes"],
            output_size=model_kwargs["g_dim"],
            num_base_filters=model_kwargs["num_base_filters"],
            num_input_channels=self.num_video_channels,
        )

        if self.stochastic:
            self.prior = MultiGaussianLSTM(
                input_size=model_kwargs["g_dim"],
                output_size=model_kwargs["z_dim"],
                hidden_size=model_kwargs["rnn_size"],
                num_layers=1,
            )
            self.posterior = MultiGaussianLSTM(
                input_size=model_kwargs["g_dim"],
                output_size=model_kwargs["z_dim"],
                hidden_size=model_kwargs["rnn_size"],
                num_layers=1,
            )
        else:
            self.prior, self.posterior = None, None

        self.decoder = ModularDecoder(
            first_block_shape=first_block_shape,
            input_size=model_kwargs["g_dim"],
            stage_sizes=model_kwargs["stage_sizes"],
            num_base_filters=model_kwargs["num_base_filters"],
            skip_type=model_kwargs["skip_type"],
            expand=model_kwargs["expand_decoder"],
            num_output_channels=self.num_video_channels,
        )

        input_size = self.get_input_size(
            model_kwargs["g_dim"], model_kwargs["action_size"], model_kwargs["z_dim"]
        )
        self.frame_predictor = MultiGaussianLSTM(
            input_size=input_size,
            output_size=model_kwargs["g_dim"],
            hidden_size=model_kwargs["rnn_size"],
            num_layers=2,
        )

        if model_kwargs.get("lecun_initialization", None):
            self.apply(init_weights_lecun)

        self.lpips = piq.LPIPS()
        # self.official_lpips = lpips_module.LPIPS(net="alex").cuda()

        self.rgb_loss_type = kwargs.get("rgb_loss_type", "l2")

        if kwargs.get("depth_predictor", None):
            self.has_depth_predictor = True
            self.depth_predictor_cfg = kwargs["depth_predictor"]
            self.load_depth_predictor()
        else:
            self.has_depth_predictor = False

        if kwargs.get("normal_predictor", None):
            self.has_normal_predictor = True
            self.normal_predictor_cfg = kwargs["normal_predictor"]
            self.load_normal_predictor()
        else:
            self.has_normal_predictor = False

        if kwargs.get("policy_networks", None):
            self.policy_feature_metric = True
            self.policy_networks_cfg = kwargs["policy_networks"]
            self.load_policy_networks()
        else:
            self.policy_feature_metric = False

    def load_policy_networks(self):
        layer = self.policy_networks_cfg["layer"]
        paths = self.policy_networks_cfg["pretrained_weight_paths"]
        self.policy_network_losses = nn.ModuleList(
            [PolicyFeatureL2Metric(path, layer) for path in paths]
        )

    def load_normal_predictor(self):
        from fitvid.scripts.train_surface_normal_model import ConvPredictor

        self.normal_predictor = ConvPredictor()
        self.normal_predictor.load_state_dict(
            torch.load(self.normal_predictor_cfg["pretrained_weight_path"])
        )
        for param in self.normal_predictor.parameters():
            param.requires_grad = False

    def load_depth_predictor(self):
        self.depth_predictor = DepthPredictor(**self.depth_predictor_cfg)

    def setup_train_losses(self):
        if self.rgb_loss_type == "l2":
            self.rgb_loss = nn.MSELoss()
        elif self.rgb_loss_type == "l1":
            self.rgb_loss = nn.L1Loss()

        if self.config.get("corr_wise", None):
            from fitvid.utils.corrwise_loss import CorrWiseLoss

            self.rgb_loss = CorrWiseLoss(
                self.rgb_loss,
                backward_warp=True,
                return_warped=False,
                padding_mode="reflection",
                scale_clip=0.1,
            )

    def get_input(self, hidden, action, z, grasped=None):
        # print("hidden, action, z, grasped:", hidden.shape, action.shape, z.shape, grasped.shape)
        inp = [hidden]
        if self.action_conditioned:
            inp += [action]
        if self.stochastic:
            inp += [z]
        if grasped is not None:
            inp += [grasped]
        # for j in range(len(inp)):
        #     print("---type", type(inp[j]))
        #     print("---", inp[j].shape)
        # input()
        return torch.cat(inp, dim=1)

    def get_input_size(self, hidden_size, action_size, z_size, grasped_size=None):
        inp = hidden_size
        if self.action_conditioned:
            inp += action_size
        if self.stochastic:
            inp += z_size
        if grasped_size is not None:
            inp += grasped_size
        return inp

    def kl_divergence(self, mean1, logvar1, mean2, logvar2, batch_size):
        kld = 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + torch.square(mean1 - mean2) * torch.exp(-logvar2)
        )
        return torch.sum(kld) / batch_size

    def compute_metrics(self, vid1, vid2, segmentation=None):
        if not (torch.all(torch.isfinite(vid1)) and torch.all(torch.isfinite(vid2))):
            return dict()

        with torch.no_grad():
            metrics = {
                "metrics/psnr": psnr(vid1, vid2),
                "metrics/lpips": lpips(self.lpips, vid1, vid2),
                "metrics/tv": tv(vid1),
                "metrics/ssim": ssim(vid1, vid2),
                "metrics/fvd": fvd(vid1, vid2),
            }

            if segmentation is not None:
                per_sample_segmented_mse = pixel_wise_loss_segmented(
                    vid1,
                    vid2,
                    segmentation,
                    loss=self.rgb_loss_type,
                    reduce_batch=False,
                )
                metrics["metrics/segmented_mse"] = per_sample_segmented_mse.mean()
                metrics[
                    "metrics/segmented_mse_per_sample"
                ] = per_sample_segmented_mse.detach()

            if self.policy_feature_metric:
                for i, policy_feature_metric in enumerate(self.policy_network_losses):
                    action_mse, feature_mse = policy_feature_metric(vid1, vid2)
                    metrics.update(
                        {
                            f"metrics/action_{i}_mse": action_mse,
                            f"metrics/policy_{i}_feature_mse": feature_mse,
                        }
                    )
            return metrics

    def forward(
        self,
        video,
        actions,
        grasped,
        segmentation=None,
        depth=None,
        normal=None,
        compute_metrics=False,
    ):
        batch_size, video_len = video.shape[0], video.shape[1]
        # print("11video.shape: ", video.shape)
        video = video.view(
            (batch_size * video_len,) + video.shape[2:]
        )  # collapse first two dims
        # print("22video.shape: ", video.shape)
        hidden, skips = self.encoder(video)
        # print("11hidden, skips: ", hidden.shape)
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        # print("22hidden, skips: ", hidden.shape)
        video = video.view(
            (
                batch_size,
                video_len,
            )
            + video.shape[1:]
        )  # reconstruct first two dims
        # print("33video.shape: ", video.shape)

        grasped_gt = grasped

        skips = {
            k: skips[k].view(
                (
                    batch_size,
                    video_len,
                )
                + tuple(skips[k].shape[1:])
            )[:, self.n_past - 1]
            for k in skips.keys()
        }
        preds, kld, means, logvars, grasped_preds = self.predict_rgb(video, actions, grasped, hidden, skips)
        # print("grasped_preds, grasped_gt: ", grasped_preds.shape, grasped_gt.shape)
        loss, preds, metrics = self.compute_loss(
            preds,
            video,
            kld,
            grasped_preds,
            grasped_gt,
            segmentation=segmentation,
            depth=depth,
            normal=normal,
            compute_metrics=compute_metrics,
        )
        metrics.update(
            {
                "hist/mean": means,
                "hist/logvars": logvars,
            }
        )
        return loss, preds, metrics

    def compute_seg_loss(self, preds, target):
        print("preds, target shapes: ", preds.shape, target.shape)
        criterion = nn.CrossEntropyLoss()
        bs, seq_len, h, w = preds.shape[0], preds.shape[1], preds.shape[3], preds.shape[4]
        # convert target from one-hot to seg value
        target = target.permute(0,1,3,4,2)
        target_seg = torch.argmax(target, axis=-1)

        # target_seg = torch.zeros((bs, seq_len, h, w), dtype=int, device=torch.device("cuda"))
        # for b in range(bs):
        #     for s in range(seq_len):
        #         for i in range(h):
        #             for j in range(w):
        #                 gt_label = torch.where(target[b, s, i, j] == 1.0)
        #                 gt_label = gt_label[0][0]
        #                 target_seg[b, s, i, j] = gt_label 
        
        seg_loss = 0
        seg_loss_per_sample = []
        for t in range(seq_len):
            seg_loss += criterion(preds[:, t], target_seg[:, t])
            seg_loss_per_sample.append(seg_loss)
            print("seg_loss: ", seg_loss)
        seg_loss /= seq_len  # Average loss over sequence length
        seg_loss_per_sample = torch.Tensor(seg_loss_per_sample)
        return seg_loss, seg_loss_per_sample
    
    def compute_loss(
        self,
        preds,
        video,
        kld,
        grasped_preds,
        grasped_gt,
        segmentation=None,
        depth=None,
        normal=None,
        compute_metrics=False,
    ):
        total_loss = 0
        metrics = dict()
        preds = dict(rgb=preds)
        for loss, weight in self.loss_weights.items():
            # print("loss, weight: ", loss, weight)
            # added by Arpit
            if loss == 'grasped':
                grasped_loss = nn.BCEWithLogitsLoss()(grasped_preds, grasped_gt[:, 1:])
                total_loss += weight * grasped_loss
                metrics["loss/grasped"] = grasped_loss
            elif loss == "kld":
                total_loss += weight * kld
                metrics["loss/kld"] = kld
            elif loss == "gripper_object_segmentation":
                seg_loss, seg_loss_per_sample = self.compute_seg_loss(
                        preds["rgb"],
                        video[:, 1:],
                    )
                total_loss += seg_loss * weight
                metrics["loss/seg_loss_per_sample"] = seg_loss_per_sample
                metrics["loss/seg_loss"] = seg_loss.detach()
                # print("---", metrics["loss/seg_loss"], metrics["loss/seg_loss_per_sample"])
            elif loss == "rgb":
                # initialize mask to be a torch tensor of all ones with same shape as video
                with torch.no_grad():
                    mse_per_sample = pixel_wise_loss(
                        preds["rgb"],
                        video[:, 1:],
                        loss="l2",
                        reduce_batch=False,
                        mask=None,
                    )
                    l1_per_sample = pixel_wise_loss(
                        preds["rgb"],
                        video[:, 1:],
                        loss="l1",
                        reduce_batch=False,
                        mask=None,
                    )
                total_loss += self.rgb_loss(preds["rgb"], video[:, 1:]) * weight
                metrics["loss/mse"] = mse_per_sample.mean().detach()
                metrics["loss/mse_per_sample"] = mse_per_sample.detach()
                metrics["loss/l1_loss"] = l1_per_sample.mean().detach()
                metrics["loss/l1_loss_per_sample"] = l1_per_sample.detach()
            elif loss == "segmented_object":
                if weight > 0:
                    segmented_mse_per_sample = pixel_wise_loss_segmented(
                        preds["rgb"],
                        video[:, 1:],
                        segmentation[:, 1:],
                        loss=self.rgb_loss_type,
                        reduce_batch=False,
                    )
                    total_loss += segmented_mse_per_sample.mean() * weight
                    metrics["loss/segmented_mse"] = segmented_mse_per_sample.mean()
                    metrics[
                        "loss/segmented_mse_per_sample"
                    ] = segmented_mse_per_sample.detach()
            elif loss == "tv":
                if weight != 0:
                    tv_loss = tv(preds)
                    total_loss += weight * tv_loss
                    metrics["loss/tv"] = tv_loss
            elif loss == "lpips":
                if weight != 0:
                    lpips_loss = lpips(self.lpips, preds["rgb"], video[:, 1:])
                    total_loss += weight * lpips_loss
                    metrics["loss/lpips"] = lpips_loss
            elif loss == "policy":
                if weight != 0 and self.policy_feature_metric:
                    feature_losses = []
                    for policy_feature_metric in self.policy_network_losses:
                        action_mse, feature_mse = policy_feature_metric(
                            preds["rgb"], video[:, 1:]
                        )
                        feature_losses.append(feature_mse)
                    feature_mse = torch.stack(feature_losses).mean()
                    metrics["loss/policy_feature_loss"] = feature_mse
                    total_loss = total_loss + weight * feature_mse
            elif loss == "depth":
                if self.has_depth_predictor:
                    if weight != 0:
                        depth_preds = self.depth_predictor(preds["rgb"])
                        depth_loss_per_sample = self.depth_predictor.depth_loss(
                            depth_preds, depth[:, 1:], reduce_batch=False
                        )
                        depth_loss = depth_loss_per_sample.mean()
                        total_loss = total_loss + weight * depth_loss
                    else:
                        with torch.no_grad():
                            depth_preds = self.depth_predictor(preds["rgb"])
                            depth_loss_per_sample = self.depth_predictor.depth_loss(
                                depth_preds, depth[:, 1:], reduce_batch=False
                            )
                            depth_loss = depth_loss_per_sample.mean()
                    preds["depth"] = depth_preds
                    metrics["loss/depth_loss"] = depth_loss
                    metrics[
                        "loss/depth_loss_per_sample"
                    ] = depth_loss_per_sample.detach()
                elif weight != 0:
                    raise ValueError(
                        "Trying to use positive depth weight but no depth predictor!"
                    )
            elif loss == "normal":
                if self.has_normal_predictor:
                    if weight != 0:
                        normal_preds = self.normal_predictor(preds["rgb"])
                        normal_loss_per_sample = mse_loss(
                            normal_preds, normal[:, 1:], reduce_batch=False
                        )
                        normal_loss = normal_loss_per_sample.mean()
                        total_loss = total_loss + weight * normal_loss
                    else:
                        with torch.no_grad():
                            normal_preds = self.normal_predictor(preds["rgb"])
                            normal_loss_per_sample = mse_loss(
                                normal_preds, normal[:, 1:], reduce_batch=False
                            )
                            normal_loss = normal_loss_per_sample.mean()
                    preds["normal"] = normal_preds
                    metrics["loss/normal_loss"] = normal_loss
                    metrics[
                        "loss/normal_loss_per_sample"
                    ] = normal_loss_per_sample.detach()
            else:
                raise NotImplementedError(f"Loss {loss} not implemented!")

        # Metrics
        metrics.update(
            {
                "loss/all": total_loss,
            }
        )
        if compute_metrics:
            if segmentation is not None:
                metrics.update(
                    self.compute_metrics(
                        preds["rgb"], video[:, 1:], segmentation[:, 1:]
                    )
                )
            else:
                metrics.update(self.compute_metrics(preds["rgb"], video[:, 1:]))

        return total_loss, preds, metrics

    def predict_rgb(self, video, actions, grasped, hidden, skips):
        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state = post_state = prior_state = None

        kld, means, logvars = torch.tensor(0).to(video), [], []
        # training
        h_preds = []
        if self.training and self.multistep:
            # print("111111111111111111111111111111111111111111111111111111111111111111")
            preds = []
            grasped_preds = []
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                if i > self.n_past:
                    # print("pred.shape: ", pred.shape)
                    # convert the predicted image (0.01, 0.09, 0.02, ...) to one-hot segmentation image (0, 1, 0, ...)
                    pred = pred.permute(0,2,3,1)
                    logSoftmax = torch.nn.LogSoftmax(dim=-1)
                    out = logSoftmax(pred)
                    inds = torch.argmax(out, axis=-1)
                    pred_seg_img = torch.nn.functional.one_hot(inds, num_classes=pred.shape[-1]).type(torch.cuda.FloatTensor)
                    print("pred_seg_img.shape: ", pred_seg_img.shape)
                    pred = pred_seg_img.permute(0,3,1,2)
                    print("pred.shape: ", pred.shape)
                    
                    h, _ = self.encoder(pred)
                if self.stochastic:
                    (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                    (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                else:
                    z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)                # print("shapes, h, actions, z_t: ", h.shape, actions.shape, z_t.shape)
                if self.use_grasped:
                    inp = self.get_input(h, actions[:, i - 1], z_t, grasped[:, i - 1])
                else:
                    inp = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
                # print("h_pred: ", h_pred.shape)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
                if self.stochastic:
                    means.append(mu)
                    logvars.append(logvar)
                    kld += self.kl_divergence(
                        mu, logvar, prior_mu, prior_logvar, batch_size
                    )
                pred = self.decoder(h_pred, skips, has_time_dim=False)
                # print("-------- Start ----------")
                # for b in range(5):
                #     print(f"{b} element in batch 1. h_pred: ", h_pred[b, :5])
                # print("------- End ------")
                if self.use_grasped:
                    grasped_pred = self.grasped_fcn(h_pred)
                # print("pred: ", pred.shape)
                preds.append(pred)
                if self.use_grasped:
                    grasped_preds.append(grasped_pred)
            preds = torch.stack(preds, axis=1)
            if self.use_grasped:
                grasped_preds = torch.stack(grasped_preds, axis=1)
            # remove later
            # print("-----------Start----------")
            # print('grasped_preds: ', grasped_preds)
            # print("-----------End----------")
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(3,3)
            # temp1 = (preds[0] * 255).to(torch.uint8).cpu().permute(0,2,3,1)
            # temp2 = (preds[1] * 255).to(torch.uint8).cpu().permute(0,2,3,1)
            # temp3 = (preds[2] * 255).to(torch.uint8).cpu().permute(0,2,3,1)
            # temp4 = (preds[3] * 255).to(torch.uint8).cpu().permute(0,2,3,1)
            # temp5 = (preds[4] * 255).to(torch.uint8).cpu().permute(0,2,3,1)
            # ax[0][0].imshow(temp1[1])
            # ax[0][1].imshow(temp1[3])
            # ax[0][2].imshow(temp1[5])
            # ax[1][0].imshow(temp2[1])
            # ax[1][1].imshow(temp2[3])
            # ax[1][2].imshow(temp2[5])
            # ax[2][0].imshow(temp3[1])
            # ax[2][1].imshow(temp3[3])
            # ax[2][2].imshow(temp3[5])
            # plt.show()
            # print("preds, grasped_preds: ", preds.shape, grasped_preds.shape)
        else:
            # print("2222222222222222222222222222222222222222222222222222222222222")
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                if self.stochastic:
                    (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                    (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                else:
                    z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)   
                if self.use_grasped:          
                    inp = self.get_input(h, actions[:, i - 1], z_t, grasped[:, i - 1])
                else:
                    inp = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
                if self.stochastic:
                    means.append(mu)
                    logvars.append(logvar)
                    kld += self.kl_divergence(
                        mu, logvar, prior_mu, prior_logvar, batch_size
                    )
            h_preds = torch.stack(h_preds, axis=1)
            preds = self.decoder(h_preds, skips)
            if self.use_grasped:
                grasped_preds = self.grasped_fcn(h_preds)
            else:
                grasped_preds = []

        if self.stochastic:
            means = torch.stack(means, axis=1)
            logvars = torch.stack(logvars, axis=1)
        else:
            means, logvars = torch.zeros(h.shape[0], video_len - 1, 1).to(
                h
            ), torch.zeros(h.shape[0], video_len - 1, 1).to(h)
        return preds, kld, means, logvars, grasped_preds

    def evaluate(self, batch, compute_metrics=False):
        ag_metrics, ag_preds = self._evaluate(
            batch, compute_metrics, autoregressive=True
        )
        # non_ag_metrics, non_ag_preds = self._evaluate(
        #     batch, compute_metrics, autoregressive=False
        # )
        ag_metrics = {f"ag/{k}": v for k, v in ag_metrics.items()}
        # non_ag_metrics = {f"non_ag/{k}": v for k, v in non_ag_metrics.items()}
        metrics = {**ag_metrics} # **non_ag_metrics
        return metrics, dict(ag=ag_preds) # non_ag=non_ag_preds

    def _evaluate(self, batch, compute_metrics=False, autoregressive=True):
        """Predict the full video conditioned on the first self.n_past frames."""
        video, actions, segmentation, grasped = (
            batch["video"],
            batch["actions"],
            batch.get("segmentation", None),
            batch["grasped"]
        )
        # print("GT grasped: ", grasped)
        # print("in evaluate: video, grasped, actions", video.shape, grasped.shape, actions.shape)
        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state = prior_state = post_state = None
        video = video.view(
            (batch_size * video_len,) + video.shape[2:]
        )  # collapse first two dims
        # print("video.shape: ", video.shape)
        hidden, skips = self.encoder(video)
        # print("hidden, skips: ", hidden.shape, skips.keys())
        # print("self.n_past: ", self.n_past)
        skips = {
            k: skips[k].view(
                (
                    batch_size,
                    video_len,
                )
                + tuple(skips[k].shape[1:])
            )[:, self.n_past - 1]
            for k in skips.keys()
        }
        # print("skpis: ", skips.keys())
        # evaluating
        preds = []
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        # print("self.n_past:", self.n_past)
        if autoregressive:
            for i in range(1, video_len):
                h, _ = hidden[:, i - 1], hidden[:, i]
                if i > self.n_past:

                    # convert the predicted image to segmentation image again
                    pred = pred.permute(0,2,3,1)
                    logSoftmax = torch.nn.LogSoftmax(dim=-1)
                    out = logSoftmax(pred)
                    inds = torch.argmax(out, axis=-1)
                    pred_seg_img = torch.nn.functional.one_hot(inds, num_classes=pred.shape[-1]).type(torch.cuda.FloatTensor)
                    print("pred_seg_img.shape: ", pred_seg_img.shape)
                    pred = pred_seg_img.permute(0,3,1,2)
                    print("pred.shape: ", pred.shape)

                    h, _ = self.encoder(pred)
                    # grasped_state = torch.round()
                if self.stochastic:
                    (z_t, prior_mu, prior_logvar), prior_state = self.prior(
                        h, prior_state
                    )
                else:
                    z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)
                # print("actions shapeee: ", actions.shape)
                # print("to make inp: ", h.shape, actions[:, i - 1].shape, z_t.shape)
                inp = self.get_input(h, actions[:, i - 1], z_t)
                # print("inp.shapeeeeee: ", inp.shape)
                # print("input to the frame predictor: ", inp.shape, pred_state)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
                # print("h_pred shape: ", h_pred.shape)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                # print("h_pred shape: ", h_pred.shape)
                pred = self.decoder(h_pred[None, :], skips)[0]
                # print("pred shape: ", pred.shape)
                preds.append(pred)
            preds = torch.stack(preds, axis=1)
        else:
            h_preds = []
            kld = torch.tensor(0).to(video)
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                if self.stochastic:
                    (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                    (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                else:
                    z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)                
                inp = self.get_input(h, actions[:, i - 1], z_t)
                # print("2inp.shapeeeeee: ", inp.shape)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
                if self.stochastic:
                    kld += self.kl_divergence(
                        mu, logvar, prior_mu, prior_logvar, batch_size
                    )
            h_preds = torch.stack(h_preds, axis=1)
            preds = self.decoder(h_preds, skips)


        video = video.view(
            (
                batch_size,
                video_len,
            )
            + video.shape[1:]
        )  # reconstuct first two dims
        # mse_per_sample = pixel_wise_loss(
        #     preds, video[:, 1:], loss="l2", reduce_batch=False
        # )
        # mse = mse_per_sample.mean()
        # l1_loss_per_sample = pixel_wise_loss(
        #     preds, video[:, 1:], loss="l1", reduce_batch=False
        # )
        # l1_loss = l1_loss_per_sample.mean()
        # metrics = {
        #     "loss/mse": mse,
        #     "loss/mse_per_sample": mse_per_sample,
        #     "loss/l1_loss": l1_loss,
        #     "loss/l1_loss_per_sample": l1_loss_per_sample,
        # }

        print("preds, video: ", preds.shape, video.shape)
        seg_loss, seg_loss_per_sample = self.compute_seg_loss(
                preds,
                video[:, 1:],
            )
        metrics = {}
        metrics["loss/seg_loss_per_sample"] = seg_loss_per_sample
        metrics["loss/seg_loss"] = seg_loss.detach()

        if compute_metrics:
            if segmentation is not None:
                metrics.update(
                    self.compute_metrics(preds, video[:, 1:], segmentation[:, 1:])
                )
            else:
                metrics.update(self.compute_metrics(preds, video[:, 1:]))

        preds = dict(rgb=preds)

        if self.has_depth_predictor:
            with torch.no_grad():
                depth_preds = self.depth_predictor(preds["rgb"], time_axis=True)
                depth_video = batch["depth_video"]
                depth_loss_per_sample = self.depth_predictor.depth_loss(
                    depth_preds, depth_video[:, 1:], reduce_batch=False
                )
                depth_loss = depth_loss_per_sample.mean()
                metrics.update(
                    {
                        "loss/depth_loss": depth_loss,
                        "loss/depth_loss_per_sample": depth_loss_per_sample,
                    }
                )

            preds["depth"] = depth_preds

        if self.has_normal_predictor:
            with torch.no_grad():
                normal_preds = self.normal_predictor(preds["rgb"], time_axis=True)
                normal_video = batch["normal"]
                normal_loss_per_sample = mse_loss(
                    normal_preds, normal_video[:, 1:], reduce_batch=False
                )
                normal_loss = normal_loss_per_sample.mean()
                metrics.update(
                    {
                        "loss/normal_loss": normal_loss,
                        "loss/normal_loss_per_sample": normal_loss_per_sample,
                    }
                )
            preds["normal"] = normal_preds
        return metrics, preds

    def test(self, batch):
        """Predict the full video conditioned on the first self.n_past frames."""
        video, actions, grasped = batch["video"], batch["actions"], batch["grasped"]
        # print("self.n_past: ", self.n_past)
        # print("video, actions, grasped: ", video.shape, actions.shape, grasped.shape)
        batch_size, video_len = video.shape[0], video.shape[1]
        action_len = actions.shape[1]
        pred_state = prior_state = None
        video = video.view(
            (batch_size * video_len,) + video.shape[2:]
        )  # collapse first two dims
        hidden, skips = self.encoder(video)
        skips = {
            k: skips[k].view(
                (
                    batch_size,
                    video_len,
                )
                + tuple(skips[k].shape[1:])
            )[:, self.n_past - 1]
            for k in skips.keys()
        }
        # evaluating
        preds = []
        grasped_preds = []
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        for i in range(1, action_len + 1):
            if i <= self.n_past:
                h = hidden[:, i - 1]
                g = grasped[:, i - 1]
                # print("11 g.shape: ", g.shape)
            if i > self.n_past:
                h, _ = self.encoder(pred)
                # g = grasped[:, i - 1]
                # print("22grasped_pred: ", grasped_pred.shape, grasped_pred[:20])
                g = torch.round(grasped_pred)
                # print("22g: ", g.shape, g[:20])

            if self.stochastic:
                (z_t, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
            else:
                z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)
            inp = self.get_input(h, actions[:, i - 1], z_t, g)
            (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
            h_pred = torch.sigmoid(h_pred)  # TODO notice
            pred = self.decoder(h_pred[None, :], skips)[0]
            grasped_pred = self.grasped_fcn(h_pred)
            grasped_pred = torch.sigmoid(grasped_pred) 
            
            preds.append(pred)
            grasped_preds.append(grasped_pred)
        preds = torch.stack(preds, axis=1)
        grasped_preds = torch.stack(grasped_preds, axis=1)
        
        video = video.view(
            (
                batch_size,
                video_len,
            )
            + video.shape[1:]
        )  # reconstuct first two dims
        return preds, grasped_preds

    def load_parameters(self, path):
        # load everything
        state_dict = torch.load(path)
        # print("state_dict: ", state_dict.keys())
        if "module.encoder.stages.0.0.Conv_0.weight" in state_dict:
            new_state_dict = {
                k[7:]: v for k, v in state_dict.items() if k[:7] == "module."
            }
            state_dict = new_state_dict
        self.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint {path}")
        if self.has_depth_predictor:
            self.load_depth_predictor()  # reload pretrained depth model
            print("Reloaded depth model")
        if self.has_normal_predictor:
            self.load_normal_predictor()
            print("Reloaded normal model")
        if self.policy_feature_metric:
            self.load_policy_networks()
            print("Reloaded policy networks")

    def state_dict(self, *args, **kwargs):
        prefixes_to_remove = ["rgb_loss"]
        # get superclass state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # create another state dict without items having prefixes in prefixes_to_remove
        new_state_dict = dict()
        for k, v in state_dict.items():
            if not any(prefix in k for prefix in prefixes_to_remove):
                new_state_dict[k] = v
        return new_state_dict
