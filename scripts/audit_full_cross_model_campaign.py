"""Independent cardinality gate for the full π0-equivalent cross-model campaign.

This audit deliberately has no dependency on the pilot campaign's producers,
summaries, completion predicates, or self-declared sample counts.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from uuid import UUID

import numpy as np


CONFIGURATIONS = (
    "pi05", "pi_fast", "internvla_a15_standard", "internvla_a15_optimized",
    "molmoact2", "groot_n17", "cosmos3_nano", "cosmos3_nano_video",
    "cosmos3_edge", "cosmos3_edge_video", "xvla", "uva", "uva_joint",
    "smolvla", "openvla_oft", "dexvla", "lingbot_vla2", "univla_baai",
    "spatialvla", "rdt_1b", "cogact_small", "openhelix", "wsa_base",
    "vitra_vla_3b", "opendw_dw05", "opendw_dw05_joint",
    "openwam_alpha_robotwin", "openwam_alpha_video", "xiaomi_robotics_1",
)
SESSIONS = 5
WARMUPS = 20
TIMED_CALLS = 200
ARRANGEMENTS_PER_IMPLEMENTATION = 23
POLICY_MODES = ("native", "batch_invariant")
SERVING_MODES = (
    "native_dynamic", "batch_invariant_dynamic", "serial_singleton", "fixed_size_padded"
)
LOADS = ("underload", "moderate", "near_saturation", "saturation")
OPERATOR_SEEDS = 100
NO_MATCHED_EVALUATOR = {"pi05", "dexvla", "vitra_vla_3b"}
PI_FAST_TOKENIZER_PROVENANCE_SHA256 = "eb27b21f5244e4b5ee164476349019f7ebd2cc8cf770bbbd781d909be41ceb09"
UVA_LIBERO10_CHECKPOINT_SHA256 = "4554b9bb2f4f3d44a7e51d5c7bd42a7740451874ac6cf1fea91e088a949f0dbf"
UVA_REPOSITORY_COMMIT = "fe3dec81af565fbb2d686072722f12340be31293"
UVA_EVALUATOR_SHA256 = "566937e82f1aa8ed3b9b582a2a70cb5a0721e3102eb3f177c60c434b89808f9c"
UVA_POLICY_SHA256 = "aefc1acc772d5566f48c93390054d868eeda310b7288d1aac0c95d0b57079129"
UVA_LIBERO_RUNNER_SHA256 = "33e8d112120d253060166a5c0e1982c5b08dc7ea0e50845c6963c1017bfac7c0"
UVA_LIBERO_ENVIRONMENT_SHA256 = "868fee8b95a21d2c0c35ba7a836484b844e9db1cfdffef135b0d3825ab960df9"
UVA_ROTATION_CONVERSIONS_SHA256 = "93735af31713ecf526cd4fb582a8914cf80bb242f4fb4d3ca82e35b6afd4d9ea"
UVA_PACKAGES = {
    "transformers": "4.28.0", "tokenizers": "0.13.3",
    "huggingface_hub": "0.25.2", "zarr": "2.16.1",
    "numcodecs": "0.11.0", "gym": "0.21.0", "kornia": "0.8.0",
}
OPENVLA_OFT_INDEX_SHA256 = "ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659"
OPENVLA_OFT_CHECKPOINT_HASHES = {
    "model-00001-of-00004.safetensors": "2809bd7be9422315c5ecbe91eea612f5b02925f37e0051728ad45bc993c79251",
    "model-00002-of-00004.safetensors": "a00a7c5f2b6586ccfc89c693a9c36f3552ff455ff2b1bfea3e92642e4cd2b6d3",
    "model-00003-of-00004.safetensors": "a894b7230a08b471af55c57dd7385fd3b51fae2c4d9307a36a8017ef57abf22a",
    "model-00004-of-00004.safetensors": "a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60",
    "action_head--150000_checkpoint.pt": "809858636cf0a65009dd567d2f4e116249442790f02b8fe31f24500ea6118908",
    "proprio_projector--150000_checkpoint.pt": "438d28e81e125166d0771762424aaf017de3a8daaebde06a5cb71157e62b3bf3",
    "config.json": "bcb688a66f3e94a42311b77d37b7e1484886701f7b2aa8e36d165dfccac92ae6",
    "dataset_statistics.json": "3f8b7854c8634ea6ccfd6ef6b8f983e3b30b067c2cdaf690b152c0899d5d9282",
}
OPENVLA_OFT_TASK_DESCRIPTIONS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]
WSA_CHECKPOINT_SHA256 = "603eacbb1a7da80dd2d41cbbd814a833bd57dec30601da5fda35651d6dd46e67"
WSA_ARTIFACT_HASHES = {
    "/workspace/assets/wsa/checkpoint/model.safetensors": WSA_CHECKPOINT_SHA256,
    "/workspace/assets/wsa/checkpoint/config.json": "e0f3e9043e97bf84569c8f430a3e666417f00d454286e83289ef010e5e507b7e",
    "/workspace/assets/wsa/checkpoint/stats.json": "5f6b2728c416bf63908a802aef8c96885d31c6fbdd7e625ec1ca84d306361f3a",
    "/workspace/assets/wsa/checkpoint/train_config.json": "16d21b3fd4fa49601cf3bb31c8176f42b8dc14834ceb04d4d84dad2ffa48e548",
    "/workspace/assets/wsa/cosmos_tokenizer/encoder.jit": "84e4c12c10568faff20da744b51c45a6b1e4385af86e80ae56bf3d1b13cda413",
    "/workspace/assets/wsa/cosmos_tokenizer/decoder.jit": "8e50c647a04decc7e939670ef7b51bc0500e0ffb4836a306a93100b0b2e957de",
    "/workspace/assets/wsa/cosmos_tokenizer/config.json": "ebf911575616266497325c38329b4212b2dfe223a6ad6c9b578fcbd7f7552aa3",
    "/workspace/assets/wsa/cosmos_tokenizer/model_config.yaml": "69bc6fabb030be2ddba76c0a6944882348c3db0426baf31ced29ca02aeaba679",
    "/workspace/assets/groot_n17/processor/config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "/workspace/assets/groot_n17/processor/merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "/workspace/assets/groot_n17/processor/preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "/workspace/assets/groot_n17/processor/tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "/workspace/assets/groot_n17/processor/tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "/workspace/assets/groot_n17/processor/video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "/workspace/assets/groot_n17/processor/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
WSA_TASK_DESCRIPTIONS = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]
UVA_TASK_DESCRIPTIONS = WSA_TASK_DESCRIPTIONS
RDT_CHECKPOINT_REPOSITORY_REVISION = "9622afab2b7ce2312a6cf1febc526928589b77eb"
RDT_ARTIFACT_HASHES = {
    "/workspace/assets/rdt/checkpoint/rdt/mp_rank_00_model_states.pt": "3fa89cbf7bbb1b2559b29f9ae153b431541cdaa0b3f1ee44b83b9934599b9fb8",
    "/workspace/assets/rdt/checkpoint/lang_embeds/text_embed_PegInsertionSide-v1.pt": "da798aab9df67a230259095c0d087772d10ff2dd089bd71762a1b6d1696dbbf3",
    "/workspace/assets/rdt/checkpoint/lang_embeds/text_embed_PickCube-v1.pt": "a9ac30bc88ef12b171e9323b8a2c64f0b0ae9a54715deb061c021deaadb37f7a",
    "/workspace/assets/rdt/checkpoint/lang_embeds/text_embed_PlugCharger-v1.pt": "c38f80c549ec1d48f74ffdeafff8dd9f02ca64de8c3bddd7fe3cca173925922d",
    "/workspace/assets/rdt/checkpoint/lang_embeds/text_embed_PushCube-v1.pt": "536272882840dd1d2aef759b7551da1515b9c0cbe45e491a025169efadf51471",
    "/workspace/assets/rdt/checkpoint/lang_embeds/text_embed_StackCube-v1.pt": "ca0a1f758d183e4c541b412efd2ffafacae7bdc7220d03f7e9dca3e93bb28985",
    "/workspace/assets/rdt/siglip/model.safetensors": "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a",
    "/workspace/assets/rdt/siglip/config.json": "adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b",
    "/workspace/assets/rdt/siglip/preprocessor_config.json": "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8",
}
SPATIALVLA_CHECKPOINT = Path("/workspace/assets/spatialvla/checkpoint")
SPATIALVLA_ARTIFACT_HASHES = {
    "model-00001-of-00002.safetensors": "713ee1bd116c8f77e2f56903d15e6c3f3dff6ea6b12fc71bfcc3bd4a53a2cc2b",
    "model-00002-of-00002.safetensors": "b8c87abc1b91598ad237c31ee2286c5c783bfb9f4142696d9b1c36e62f634a34",
    "config.json": "0c5d94be86f9e15b708650c4b829033ae99243f85507012b1dafa0263f799909",
    "model.safetensors.index.json": "6c351156136b2452db86d3a426ff30adc9d11363b43f46fa69fe79d76b6d5c79",
    "processor_config.json": "8712b6d5171f74e90f951ec7b148ab90509f28434e36b2da959527fa0ddb57ea",
    "processing_spatialvla.py": "c20b15460633f65809788c633936347fada2f2bbcfffba898d2f0b4b99807b33",
    "modeling_spatialvla.py": "a04008cf364e094079493e5982fa2b32fca12874b0a42b645e0aecab024e7236",
    "action_tokenizer.py": "b6b18eee0320e77156881aea4a92ea5f54dc278d43c1ea72c3a0deaa9d505a31",
    "configuration_spatialvla.py": "63e100324daf81d2cd7b6c5e6fafcace67d660586ef7b509bb38cc8fba72c1e3",
    "tokenizer.json": "2523a63c898ebf0a32c7282a2e459ef2c950a846c5f3172305089e4149b6b6c3",
    "tokenizer_config.json": "33f38858ac90afd5ac29942110dd3b98f40e1e194fca254714bab5d88f57cc65",
    "preprocessor_config.json": "bdb79dc3255a75c1c4be8ac1ddd4e2aa08b9a29eee0431755b7b0faa3b54e413",
    "generation_config.json": "79c657e6d0660218a91c360516cae5b8fa0cf1dd447476c8dcdb815c8ff994b1",
}
COGACT_CHECKPOINT = Path("/workspace/assets/cogact_small/checkpoints/CogACT-Small.pt")
COGACT_CHECKPOINT_SHA256 = "013cf0d4954e8a560064905ee03785ce3d0fa126697d4410cfde9a6754665e44"
COGACT_CHECKPOINT_BYTES = 30_215_225_166
COGACT_TASK_MANIFEST_SHA256 = "37c70a256a17c0e11a2343140d550549e9b74ff4896059ae14f863b96f439eba"
COGACT_CLIENT_SHA256 = "afa350668d8d77b6edc9ad048706369e56a5fa4a5163abf1efa31142f38f3ba6"
COGACT_SERVER_SHA256 = "ce2a0a8c7ac0795f91a5544b5fe52e180931ddbcd6bb394f9c1894da09cfc531"
COGACT_POSE_KEYS = (
    ("tcp_pose",) * 4 + ("base_pose",) * 6 + ("tcp_pose",) * 2
)
DEXVLA_SOURCE_SHA256 = "abd1a0681df1a14e1fc09a799a7b5965e4a73bc7f233e2bbd5441303adb4203a"
DEXVLA_SOURCE_BYTES = 1_095_683_384
DEXVLA_SOURCE_REVISION = "5e7f53a7d7d5198ffe68664f0426d3e5e3d0d366"
DEXVLA_CHECKPOINT_REVISION = "f99bdd408e7dadf7ff1b960431c15781ce761b52"
DEXVLA_REPOSITORY_REVISION = "fc21a822f4c774e242eb6f1ab4a235788de7aba9"
DEXVLA_CHECKPOINT_HASHES = {
    "added_tokens.json": (392, "7fa54985b58718a8fdb4f4d97484c4bd908db114847675e4bf3afe3e1d5d7bd4"),
    "chat_template.json": (1050, "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96"),
    "config2.json": (3788, "9186f1dad06f01223abdc863d6af6a63289d3af9496202fba668a68534e50c88"),
    "dataset_stats.pkl": (11851, "7bee355ea318a692b2b263a027792bd0e56b10429f6bb3f0e3938ad109ffe533"),
    "generation_config.json": (215, "2f8fa12f569c384fb1907845882cb9fa382fdadcb25a92751ff6b220f764dadf"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model-00001-of-00002.safetensors": (4_991_948_224, "3b096461b657171a3b37585e44e2e32171d045aa3bddea0295f4d2c3971baf7a"),
    "model-00002-of-00002.safetensors": (1_832_942_772, "c5b2b1e07e55e37b0cbaa1dab222872d4ea79ddd063062b2e65c0f8a2073aa87"),
    "model.safetensors.index.json": (84884, "c118abc7c960406ead47e938276d47ec44aecb8076daaa9af1c0199c3d8b7c64"),
    "preprocessor_config.json": (347, "b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc"),
    "special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    "tokenizer.json": (11_420_538, "88a3a6fcb80132f76da8aa40cdc3fccd7e5d8468ef15421f5b0c2715e85217d2"),
    "tokenizer_config.json": (4256, "c715bebf439296530bc63ac45461f3f70d6e45b561ebf02d3a21091e07b698a1"),
    "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
LINGBOT_VLA2_REPOSITORY_REVISION = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
LINGBOT_VLA2_CHECKPOINT_REVISION = "0451855729ec904f970600e0aec8b84661423afe"
LINGBOT_VLA2_CHECKPOINT_HASHES = {
    "added_tokens.json": (707, "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
    "chat_template.jinja": (5_292, "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"),
    "config.json": (31, "7c1fddcf59379627bfafda7b5923092f370fa28f15d2132ee305a037335dfb80"),
    "model-00001-of-00006.safetensors": (4_947_507_248, "514801684e0f23aaa4bb64a4fd81e420b97e094331499852d5f2cc567750b360"),
    "model-00002-of-00006.safetensors": (4_944_315_304, "e7b00cf59b21c2f5e9f76f7687ba423c6fe24bfa9fa8e313ef346bb081e73c9f"),
    "model-00003-of-00006.safetensors": (4_944_315_360, "ace74f6d5214059b6387b5728c570a452d51b5950dd9904fe29607dbfa41447b"),
    "model-00004-of-00006.safetensors": (4_992_858_968, "1f1cb6b4099e264b202eae0244f3730e1ced910c0b34355fff2a719993d0cbe5"),
    "model-00005-of-00006.safetensors": (4_963_062_048, "1316d6fe054253650dc65bae416ae6cf9afa283aad82a9712b02b0dd7029394b"),
    "model-00006-of-00006.safetensors": (711_830_844, "0e1dfc6a23c43f3c387276e2389799867dfa90dc9727bb8e21250c73bfe7dd23"),
    "model.safetensors.index.json": (207_389, "661b9d29dc14b73adfe233d2d989517f92e561f77424a7d2097a5d735c90b57f"),
    "preprocessor_config.json": (782, "93585062a80db5e8ca038efc7726a3e6411d9db948472d81d63c6303993be8c5"),
    "special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    "tokenizer.json": (11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (5_472, "cf43a5bf1a49ee69ecced02f419b169e72559034dcf15af47cf775bd253830f0"),
    "video_preprocessor_config.json": (817, "59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1"),
    "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
XIAOMI_DATASET_REVISION = "1f57c08d41d503c7a2c645c4181bae4a6d6ab997"
XIAOMI_REPOSITORY_REVISION = "0dd7aef8dc87296246aae812a1f59ccb708e5546"
XIAOMI_CHECKPOINT_REVISION = "82b7c2210c1a39fca4a87e681cffbbed134ccde0"
XIAOMI_SOURCE_SELECTION_SHA256 = "70123ba9826795a1076e8b0ae2ec009962672af6bae399fc98d9dcaee66d5708"
XIAOMI_SOURCE_METADATA_HASHES = {
    "meta/episodes.jsonl": "695dc10872f2c750919f22150d88e0c28504bfbded93459207e4402b68ccc3f8",
    "meta/info.json": "3cce638db7949556ab3a36bcf8f0e7f06475906f939b871b78796f5f9ab6a842",
    "meta/stats.json": "ffb354d6268646b9c54f661ab7777a5b8d03bdffdabcda78a93e187f07b0617a",
    "meta/tasks.jsonl": "fdcebf3fc6206c089a35cc60ba07b64e9eec71590838a46e57560a41057e10f0",
}
XIAOMI_CHECKPOINT_HASHES = {
    "config.json": (5_580, "73a00f2dab6fb727942ec2256d78ec964332bc9a7a098e75ba580ab06040d1f0"),
    "model-00001-of-00003.safetensors": (4_990_499_880, "ae78351b0612ab4d9b22bb8f9e05e22f8ea444ee6a8e27c284bc78531848fc40"),
    "model-00002-of-00003.safetensors": (4_999_627_848, "c71009db51ea59a376085bc8940d79708dccbcff41ac79ba955b87c733147619"),
    "model-00003-of-00003.safetensors": (116_305_608, "cb67483e9e0b9ddf98c66961fc1ee38d73084cc5f2c065166db382d20b6a8197"),
    "model.safetensors.index.json": (98_622, "9663e68cab7d296c4b41f3a1885afb1f6f2722828825ad4ba6bb7768c29ae396"),
    "modeling_mibot.py": (82_265, "214effa075d42f1f938984afcac3b17fa16c08f726c5bf7049b5bd82cd26a4a2"),
    "preprocessor_config.json": (30_028, "cf2a1df1f3d4351fac24d0e5d057dc4d762f028349c543c1ac726395b098ca8f"),
    "processing_mibot.py": (19_815, "45c618e4d852d4948cec32f56fc55fb24548d4a15855e2ae2d7313db4b4d3389"),
}
VITRA_DATASET_REVISION = "fcd93409b47b01e6075b8d7ced28ecb8a0327964"
VITRA_REPOSITORY_REVISION = "b35517202b39d32a753fdd42014b2cc3c41fab58"
VITRA_CHECKPOINT_REVISION = "4bd47d5be2f664b9ae9636094a07a20e905d1ec1"
VITRA_CHECKPOINT_BYTES = 15_067_578_877
VITRA_CHECKPOINT_SHA256 = "7f57dfd4451d68f41accab8a06bb388d6fdc2cce87ee8b31fafa7c8f01d83796"
VITRA_SOURCE_BYTES = 91_473_237
VITRA_SOURCE_AGGREGATE_SHA256 = (
    "11fcb3067b16a3c7f2fccdac0c5ca799667a1ef7f4318afd6f2db57aebdd7f5f"
)
VITRA_DIAGNOSTIC_EPISODES = {0, 1, 2, 3, 4, 6, 8, 10, 12, 47}
VITRA_XHAND_HUMAN_MAPPING = (
    (10, 8, 1), (11, 14, 1), (12, 17, 1), (13, 23, 1),
    (16, 26, 1), (17, 32, 1), (14, 35, 1), (15, 41, 1),
    (7, 43, -1), (6, 44, 1), (8, 46, -1), (9, 7, 1),
)
OPENHELIX_DATASET_REVISION = "0874cd64f0df8dcc3a3534360ca448705e44e954"
OPENHELIX_ARCHIVE_BYTES = 21_474_836_480
OPENHELIX_ARCHIVE_SHA256 = "aa735af43b268677c94d2c8edb32f78586bcd129f86acd4857f793f306d1ec41"
OPENHELIX_REPOSITORY_REVISION = "d88867fa56977f401a7c89d62b9ebce87d8078f0"
OPENHELIX_CHECKPOINT_REVISION = "1a6fabbe148af045ed0710d007ed24c36fa83a53"
OPENHELIX_CHECKPOINT_FILES = [
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00000.safetensors", 2_105_645_528, "7bc815c581781b9d864ff22fbfd87f84d45adda0b283609d42f5716f6b933ac7"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00001.safetensors", 2_114_018_192, "6f015b79cea364ea356c0e11414e9dbe71f4736b2b5b8523119a34a087e0e758"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00002.safetensors", 2_114_034_864, "2ec780a890f2554385765fe3f4beedeb6b953d7eaf6f3887ad1535e56976433f"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00003.safetensors", 2_124_504_248, "6d552bc92746187c7d1614186e924dee8c3b456c06be03868da9d2ac07410128"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00004.safetensors", 2_057_395_512, "247bc9b65065446a596a3c431077bdbe4126681821334c2019369aaa65f49620"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00005.safetensors", 2_114_018_256, "4172759f8b2e57b1e0d3d8daaddea4b5489adfa546f07b4a7ce1b53db55f2e62"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00006.safetensors", 1_504_179_160, "ee2df39c2fc516148d2d82e108f66c189f2cd769fb4e42cc333e6dca23d87810"),
    ("prompt_tuning_aux/policy.pth", 445_004_793, "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638"),
]
INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256 = (
    "329cf78b3367555a0440616959ee8515e9490116530792b55db0ea3d1ef50bdf"
)
RDT_TASK_NAMES = [
    "PegInsertionSide-v1", "PickCube-v1", "StackCube-v1", "PlugCharger-v1", "PushCube-v1",
]
EXPECTED_EXECUTION_CONFIG = {
    "molmoact2": {
        "camera_features": [
            "observation.images.image", "observation.images.wrist_image",
        ],
        "source_image_shape_hwc": [256, 256, 3],
        "processed_pixel_values_shape_per_request": [2, 729, 588],
        "processed_image_token_pooling_shape_per_request": [392, 4],
        "state_dim": 8,
        "native_language_padding": "longest_in_batch_right",
        "invariant_language_padding": "max_length_right",
        "canonical_language_tokens": 497,
        "language_pad_token_id": 151643,
        "flow_steps": 10,
        "max_action_horizon": 10,
        "max_action_dim": 32,
        "robot_output_dim": 7,
        "precision": "bfloat16",
    },
    "pi05": {
        "language_padding": "max_length_right",
        "language_max_length": 200,
        "camera_features": [
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ],
        "missing_camera_contract": "official_zero_image_false_mask",
        "flow_steps": 10,
        "chunk_size": 50,
        "max_action_dim": 32,
    },
    "pi05_libero": {
        "checkpoint_revision": "7152f13e34c6c115d808aea0e5d08aca74eac14b",
        "suite": "libero_10", "language_padding": "max_length_right",
        "language_tokens": 200, "sampler": "unchanged_official_predict_action_chunk",
        "flow_steps": 10, "chunk_size": 50, "max_action_dim": 32,
        "robot_action_dim": 7, "normalization": "checkpoint_mean_std",
        "precision": "bfloat16", "tf32": False, "execution": "eager",
    },
    "rdt_maniskill_exact": {
        "checkpoint_revision": "9622afab2b7ce2312a6cf1febc526928589b77eb", "suite": "maniskill_five",
        "language_padding": "fixed_right_in_both_modes", "language_tokens": 30, "image_history": 2,
        "camera_slots": 3, "vision_encoding": "genuinely_batched_six_slots_per_request", "diffusion_steps": 5,
        "chunk_size": 64, "precision": "bfloat16", "tf32": False, "control_frequency": 25,
        "action_dim": 8, "noise_dim": 128, "sampler": "official_maniskill_wrapper_equivalent_DPM_solver", "execution": "eager",
    },
    "cogact_simpler_exact": {
        "checkpoint_revision": "ca302a300542393a0b06e291465076a7d9105c23", "suite": "simpler_google_four",
        "language_padding": "fixed_right_in_both_primary_modes", "language_tokens": 64,
        "precision": "vlm_bfloat16_diffusion_float32", "tf32": False, "chunk_size": 16, "action_dim": 7,
        "diffusion_steps": 10, "cfg_scale": 1.5, "execution": "eager",
        "released_default_fidelity": "separate_longest_padding_singleton_measurement_required",
    },
    "internvla_a15_standard": None,
    "smolvla": {
        "native_language_padding": "longest", "invariant_language_padding": "max_length",
        "language_max_length": 48, "language_pad_token_id": 2,
    },
    "groot_n17": {
        "native_language_padding": "longest_left", "invariant_language_padding": "fixed_left_156",
        "language_pad_token_id": 151643,
    },
    "internvla_a15_optimized": {
        "backend": "optimized", "optimized_graph_cache_policy": "recapture_per_policy_call",
    },
    "openvla_oft": {
        "native_language_padding": "longest_in_batch_right",
        "invariant_language_padding": "canonical_right",
        "language_max_length": 37,
        "language_pad_token_id": 32000,
    },
    "univla_baai": {
        "native_language_padding": "longest_in_batch_right",
        "invariant_language_padding": "canonical_right",
        "language_max_length": 1293,
        "language_pad_token_id": 151643,
        "maximum_generated_action_tokens": 80,
        "decoding": "greedy_constrained_fast",
        "camera_resize_hw": [200, 200],
    },
    "spatialvla": {
        "native_language_padding": "longest_in_batch_right",
        "invariant_language_padding": "canonical_right",
        "canonical_text_length": 64,
        "unnorm_key": "bridge_orig/1.0.0",
        "do_normalize": False,
        "generation": "greedy_max_new_tokens_256",
    },
    "rdt_1b": {
        "native_language_padding": "longest_in_batch_right",
        "invariant_language_padding": "canonical_right",
        "canonical_language_tokens": 30,
        "image_history": 2,
        "num_cameras": 3,
        "diffusion_steps": 5,
    },
    "cogact_small": {
        "native_language_padding": "longest_in_batch_right",
        "invariant_language_padding": "canonical_right",
        "canonical_language_tokens": 27,
        "prompt_builder": "PurePromptBuilder",
        "cfg_scale": 1.5,
        "ddim_steps": 10,
    },
    "dexvla": {
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "image_shape_hwc": [480, 640, 3],
        "state_dim": 14,
        "reasoning_generation": "greedy_max_new_tokens_256",
        "diffusion_steps": 10,
        "action_chunk_size": 50,
        "action_dim": 14,
        "source_limit_observations": 396,
    },
    "lingbot_vla2": {
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "image_shape_hwc": [480, 640, 3],
        "state_dim": 14,
        "flow_steps": 10,
        "action_chunk_size": 50,
        "model_action_dim": 55,
        "source_limit_observations": 396,
        "precision": "float32",
        "deterministic_moe": "deterministic_token_choice_moe",
    },
    "xiaomi_robotics_1": {
        "camera_features": ["image_left", "image_right", "wrist_image"],
        "source_image_shape_hwc": [128, 128, 3],
        "policy_image_shape_hwc": [256, 256, 3],
        "real_state_dim": 25,
        "padded_state_dim": 60,
        "canonical_language_tokens": 260,
        "flow_steps": 5,
        "action_chunk_size": 10,
        "model_action_dim": 60,
        "robot_output_dim": 7,
        "robot_type": "robocasa_mg",
    },
    "vitra_vla_3b": {
        "source": "microsoft/VITRA-TeleData",
        "source_episodes": 50,
        "source_image_shape_hwc": [720, 1280, 3],
        "model_image_shape_hwc": [224, 398, 3],
        "real_xhand_state_dim": 18,
        "human_statistics_state_dim": 122,
        "padded_model_state_dim": 212,
        "canonical_language_tokens": 276,
        "ddim_steps": 10,
        "action_chunk_size": 16,
        "model_action_dim": 192,
        "robot_output_dim": 102,
        "precision": "float32",
    },
    "openhelix": {
        "source": "KyujinL/CALVIN_ABC",
        "planner_image_shape_hwc": [200, 200, 3],
        "policy_cameras": ["static", "gripper"],
        "policy_rgb_shape_bchw": [2, 3, 200, 200],
        "policy_point_cloud_shape_bchw": [2, 3, 200, 200],
        "proprio_dim": 8,
        "planner_precision": "bfloat16",
        "policy_precision": "float32",
        "ddpm_steps": 25,
        "action_chunk_size": 19,
        "robot_output_dim": 7,
        "instruction": "push the sliding door to the right side",
    },
}
EXPECTED_FIXED_LANGUAGE_SHAPE = {
    "rdt_maniskill_exact": {"language_padding": "fixed_right_in_both_modes", "language_tokens": 30, "canonical_length": 30},
    "cogact_simpler_exact": {"language_padding": "fixed_right_in_both_primary_modes", "language_tokens": 64, "canonical_length": 64},
    "molmoact2": {
        "language_padding": "max_length_right",
        "language_tokens": 497,
        "canonical_length": 497,
    },
    "pi05": {"language_padding": "max_length_right", "language_tokens": 200, "canonical_length": 200},
    "pi05_libero": {"language_padding": "max_length_right", "language_tokens": 200, "canonical_length": 200},
    "smolvla": {"language_padding": "max_length_right", "language_tokens": 48, "canonical_length": 48},
    "groot_n17": {"language_padding": "fixed_left", "language_tokens": 156, "canonical_length": 156},
    "openvla_oft": {"language_padding": "canonical_right", "language_tokens": 37, "canonical_length": 37},
    "univla_baai": {"language_padding": "canonical_right", "language_tokens": 1293, "canonical_length": 1293},
    "spatialvla": {"language_padding": "max_length_right", "canonical_length": 64},
    "rdt_1b": {"language_padding": "canonical_right", "language_tokens": 30, "canonical_length": 30},
    "cogact_small": {"language_padding": "canonical_right", "language_tokens": 27, "canonical_length": 27},
    "xiaomi_robotics_1": {"language_padding": "canonical_right", "language_tokens": 260, "canonical_length": 260},
    "vitra_vla_3b": {"language_padding": "canonical_right", "language_tokens": 276, "canonical_length": 276},
}


def required_arrangements(batch_sizes=(1, 2, 4, 8)) -> set[tuple]:
    result = {("singleton", 1, "first", "none")}
    for size in batch_sizes:
        if size == 1:
            continue
        positions = ("first", "last") if size == 2 else ("first", "middle", "last")
        result.update(
            ("batch_position", size, position, companion)
            for position in positions
            for companion in ("duplicate", "diverse")
        )
        result.add(("permutation", size, "last", "diverse"))
        result.add(("partition", size, "first", "diverse"))
    if batch_sizes == (1, 2, 4, 8):
        assert len(result) == ARRANGEMENTS_PER_IMPLEMENTATION
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        advised_through = 0
        try:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                position = stream.tell()
                if position - advised_through >= 64 * 1024 * 1024:
                    try:
                        os.posix_fadvise(
                            stream.fileno(), advised_through,
                            position - advised_through, os.POSIX_FADV_DONTNEED,
                        )
                        advised_through = position
                    except (AttributeError, OSError):
                        pass
        finally:
            try:
                os.posix_fadvise(
                    stream.fileno(), advised_through, 0, os.POSIX_FADV_DONTNEED
                )
            except (AttributeError, OSError):
                pass
    return digest.hexdigest()


def is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def observation_image_sha256(row: dict) -> object:
    return row.get("image_sha256", row.get("source_image_rgb_sha256"))


def request_multiplicity(manifest: dict, issues: list[str]) -> int:
    observations = manifest.get("observations", [])
    stochastic = all(
        len(item.get("noise", [])) == 3 and len(item.get("request_ids", [])) == 3
        for item in observations
    )
    if stochastic and observations:
        return 3
    contract = manifest.get("deterministic_decoding_contract") or {}
    evidence = contract.get("evidence_path") if isinstance(contract, dict) else None
    if observations and all(len(item.get("request_ids", [])) == 1 for item in observations) and evidence and Path(evidence).is_file() and sha256(Path(evidence)) == contract.get("evidence_sha256") and contract.get("reason"):
        return 1
    issue(issues, "replay: require three stored noises/request IDs or a pinned deterministic decoding contract")
    return 0


def expected_replay_counts(model_id: str) -> tuple[int, int, int]:
    """Return hard-coded observation, diagnostic, and held-out cardinalities."""
    return (
        (396, 200, 196)
        if model_id in {"dexvla", "lingbot_vla2"}
        else (1000, 200, 800)
    )


def capacity_batch_sizes(model_id: str) -> tuple[int, ...]:
    return (8, 16, 32) if model_id == "dexvla" else (16, 32)


def replay_batch_sizes(model_id: str, capacity: dict, mode: str) -> tuple[int, ...]:
    sizes = [1, 2, 4] if model_id == "dexvla" else [1, 2, 4, 8]
    sizes.extend(
        size for size in capacity_batch_sizes(model_id)
        if capacity.get("feasible", {}).get(f"{mode}|B{size}")
    )
    return tuple(sizes)


def _dexvla_h5py():
    try:
        import h5py
    except ModuleNotFoundError:
        dependency = Path("/workspace/envs/dexvla/lib/python3.12/site-packages")
        if not dependency.is_dir():
            raise
        sys.path.insert(0, str(dependency))
        import h5py
    return h5py


def source_limited_hdf5_audit(
    model_id: str, manifest: dict, observations: list[dict], issues: list[str]
) -> None:
    """Independently prove and inspect a 396-observation public HDF5 limit."""
    parameters = {
        "dexvla": (
            "DexVLA",
            "dexvla-episode-47-frame-",
            DEXVLA_REPOSITORY_REVISION,
            DEXVLA_CHECKPOINT_REVISION,
            DEXVLA_CHECKPOINT_HASHES,
        ),
        "lingbot_vla2": (
            "LingBot-VLA2",
            "lingbot-vla2-episode-47-frame-",
            LINGBOT_VLA2_REPOSITORY_REVISION,
            LINGBOT_VLA2_CHECKPOINT_REVISION,
            LINGBOT_VLA2_CHECKPOINT_HASHES,
        ),
    }
    if model_id not in parameters:
        raise ValueError(f"unsupported source-limited HDF5 model: {model_id}")
    label, id_prefix, repository_revision, checkpoint_revision, checkpoint_hashes = (
        parameters[model_id]
    )
    start = len(issues)
    inventory_path = Path(manifest.get("source_inventory_path", ""))
    inventory = read_json(inventory_path) if inventory_path.is_file() else None
    if (not inventory or sha256(inventory_path) != manifest.get("source_inventory_sha256")
            or inventory.get("unique_public_episodes") != 1
            or inventory.get("unique_public_observations") != 396
            or len(inventory.get("archive_members", [])) != 2
            or {row.get("bytes") for row in inventory.get("archive_members", [])}
               != {DEXVLA_SOURCE_BYTES}
            or {row.get("sha256") for row in inventory.get("archive_members", [])}
               != {DEXVLA_SOURCE_SHA256}
            or inventory.get("retained_source", {}).get("observations") != 396
            or inventory.get("retained_source", {}).get("sha256") != DEXVLA_SOURCE_SHA256):
        issue(issues, f"replay: {label} inventory does not prove the one-episode/396-frame source limit")
        return
    source_limit = manifest.get("source_limit") or {}
    if (source_limit.get("unique_public_episodes") != 1
            or source_limit.get("unique_public_observations") != 396
            or source_limit.get("all_unique_observations_used") is not True
            or source_limit.get("no_observation_duplication") is not True
            or manifest.get("dataset_revision") != DEXVLA_SOURCE_REVISION
            or manifest.get("official_repository_revision") != repository_revision
            or manifest.get("checkpoint_revision") != checkpoint_revision):
        issue(issues, f"replay: {label} source/repository/checkpoint provenance differs")
    checkpoint_root = Path(manifest.get("checkpoint_root", ""))
    recorded_checkpoint = manifest.get("checkpoint_files") or {}
    expected_records = {
        name: {"bytes": size, "sha256": digest}
        for name, (size, digest) in checkpoint_hashes.items()
    }
    if recorded_checkpoint != expected_records:
        issue(issues, f"replay: {label} checkpoint manifest differs from pinned exact files")
    else:
        for name, (size, digest) in checkpoint_hashes.items():
            path = checkpoint_root / name
            if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
                issue(issues, f"replay: {label} checkpoint file changed or disappeared: {name}")
                break
    source = Path(manifest.get("source_root", "")) / manifest.get("source_file", "")
    if (not source.is_file() or source.stat().st_size != DEXVLA_SOURCE_BYTES
            or sha256(source) != DEXVLA_SOURCE_SHA256):
        issue(issues, f"replay: {label} retained HDF5 source changed or disappeared")
        return
    try:
        h5py = _dexvla_h5py()
        with h5py.File(source, "r") as episode:
            cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
            if (episode["observations/qpos"].shape != (396, 14)
                    or episode["observations/qpos"].dtype != np.float32
                    or any(episode[f"observations/images/{camera}"].shape
                           != (396, 480, 640, 3) for camera in cameras)
                    or any(episode[f"observations/images/{camera}"].dtype
                           != np.uint8 for camera in cameras)
                    or episode["language_raw"][0] != b"Cook rice."):
                issue(issues, f"replay: {label} HDF5 schema/task differs from the pinned source")
                return
            permutation = np.random.default_rng(20261015).permutation(396)
            diagnostic = {int(index) for index in permutation[:200]}
            physical_hashes = set()
            for index, row in enumerate(observations):
                images = [np.asarray(episode[f"observations/images/{camera}"][index])
                          for camera in cameras]
                qpos = np.asarray(episode["observations/qpos"][index])
                image_hashes = {
                    camera: hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                    for camera, image in zip(cameras, images)
                }
                state_hash = hashlib.sha256(np.ascontiguousarray(qpos).tobytes()).hexdigest()
                physical = hashlib.sha256()
                for image in images:
                    physical.update(np.ascontiguousarray(image).tobytes())
                physical.update(np.ascontiguousarray(qpos).tobytes())
                physical_hash = physical.hexdigest()
                expected_companions = [
                    (index + (slot * 396) // 32) % 396 for slot in range(1, 32)
                ]
                expected_images = {
                    camera: f"episode_47.hdf5::observations/images/{camera}[{index}]"
                    for camera in cameras
                }
                expected_split = "diagnostic" if index in diagnostic else "heldout"
                if (row.get("id") != f"{id_prefix}{index:03d}"
                        or row.get("replay_index") != index
                        or row.get("source_frame_index") != index
                        or row.get("task") != "Cook rice."
                        or row.get("episode_id") != "episode_47"
                        or row.get("observation_index") != index
                        or row.get("split") != expected_split
                        or not np.array_equal(np.asarray(row.get("state"), dtype=np.float32), qpos)
                        or row.get("state_sha256") != state_hash
                        or row.get("images") != expected_images
                        or row.get("image_sha256") != image_hashes
                        or row.get("physical_observation_sha256") != physical_hash
                        or row.get("companion_replay_indices") != expected_companions):
                    issue(issues, f"replay: {label} frozen source row differs at index {index}")
                    break
                physical_hashes.add(physical_hash)
            if len(physical_hashes) != 396:
                issue(issues, f"replay: {label} has only {len(physical_hashes)}/396 unique physical observations")
    except (KeyError, OSError, TypeError, ValueError, ModuleNotFoundError) as error:
        issue(issues, f"replay: {label} HDF5 source could not be independently inspected: {error}")
    if len(issues) == start and manifest.get("split_counts") != {"diagnostic": 200, "heldout": 196}:
        issue(issues, f"replay: {label} recorded split counts differ from 200/196")


def dexvla_source_limit_audit(
    manifest: dict, observations: list[dict], issues: list[str]
) -> None:
    """Compatibility wrapper for focused DexVLA audit tests."""
    source_limited_hdf5_audit("dexvla", manifest, observations, issues)


def lingbot_vla2_source_limit_audit(
    manifest: dict, observations: list[dict], issues: list[str]
) -> None:
    source_limited_hdf5_audit("lingbot_vla2", manifest, observations, issues)


def _xiaomi_parquet():
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError:
        dependency = Path("/workspace/envs/dexvla/lib/python3.12/site-packages")
        if not dependency.is_dir():
            raise
        sys.path.insert(0, str(dependency))
        import pyarrow.parquet as parquet
    return parquet


def xiaomi_source_audit(
    manifest: dict, observations: list[dict], issues: list[str]
) -> None:
    """Independently verify Xiaomi's 100-episode/1,000-real-frame freeze."""
    expected_inventory = {
        "codebase_version": "v2.0", "robot_type": "panda",
        "total_episodes": 70_513, "total_frames": 19_636_669,
        "total_tasks": 340, "total_chunks": 71, "chunks_size": 1000, "fps": 10,
    }
    if (
        manifest.get("dataset_revision") != XIAOMI_DATASET_REVISION
        or manifest.get("dataset_inventory") != expected_inventory
        or manifest.get("official_repository_revision") != XIAOMI_REPOSITORY_REVISION
        or manifest.get("checkpoint_revision") != XIAOMI_CHECKPOINT_REVISION
        or manifest.get("source_metadata") != XIAOMI_SOURCE_METADATA_HASHES
        or manifest.get("state_contract")
        != "complete real 25-D dataset state padded with zeros to the official 60-D policy input"
    ):
        issue(issues, "replay: Xiaomi dataset/repository/state provenance differs")

    source_hashes = manifest.get("source_sha256") or {}
    expected_paths = {
        f"data/chunk-000/episode_{index:06d}.parquet" for index in range(100)
    }
    selection_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if set(source_hashes) != expected_paths or selection_hash != XIAOMI_SOURCE_SELECTION_SHA256:
        issue(issues, "replay: Xiaomi source selection is not the pinned first 100 episodes")
        return

    checkpoint_root = Path(manifest.get("checkpoint_root", ""))
    checkpoint_records = manifest.get("checkpoint_files") or {}
    for name, (size, digest) in XIAOMI_CHECKPOINT_HASHES.items():
        expected = {"bytes": size, "sha256": digest}
        path = checkpoint_root / name
        if (
            checkpoint_records.get(name) != expected
            or not path.is_file()
            or path.stat().st_size != size
            or sha256(path) != digest
        ):
            issue(issues, f"replay: Xiaomi exact checkpoint differs: {name}")
            break

    source_root = Path(manifest.get("source_root", ""))
    fixture_root = Path(manifest.get("fixture_root", ""))
    metadata_paths = {
        name: source_root / name for name in XIAOMI_SOURCE_METADATA_HASHES
    }
    if any(
        not path.is_file() or sha256(path) != XIAOMI_SOURCE_METADATA_HASHES[name]
        for name, path in metadata_paths.items()
    ):
        issue(issues, "replay: Xiaomi pinned dataset metadata changed or disappeared")
        return
    try:
        episode_metadata = [
            json.loads(line)
            for line in metadata_paths["meta/episodes.jsonl"].read_text().splitlines()[:100]
        ]
        task_metadata = [
            json.loads(line)
            for line in metadata_paths["meta/tasks.jsonl"].read_text().splitlines()
            if line.strip()
        ]
        tasks = {row["task_index"]: row["task"] for row in task_metadata}
        by_task = defaultdict(list)
        for row in episode_metadata:
            by_task[row["tasks"][0]].append(row["episode_index"])
        diagnostic_episodes = {
            episode
            for episodes in by_task.values()
            for episode in episodes[: len(episodes) // 5]
        }
        if (
            [row.get("episode_index") for row in episode_metadata] != list(range(100))
            or sorted(map(len, by_task.values())) != [40, 60]
            or len(diagnostic_episodes) != 20
            or len(tasks) != 340
        ):
            issue(issues, "replay: Xiaomi first-100 episode/task metadata differs")
            return

        parquet = _xiaomi_parquet()
        physical_hashes = set()
        cameras = ("image_left", "image_right", "wrist_image")
        for episode_index, metadata in enumerate(episode_metadata):
            relative = f"data/chunk-000/episode_{episode_index:06d}.parquet"
            source = source_root / relative
            if (
                not source.is_file()
                or sha256(source) != source_hashes[relative]
            ):
                issue(issues, f"replay: Xiaomi source episode changed: {episode_index}")
                break
            table = parquet.read_table(source)
            indexes = [
                int(value) for value in np.linspace(0, table.num_rows - 1, 10, dtype=np.int64)
            ]
            selected = table.take(indexes).to_pylist()
            for local_index, (frame_index, source_row) in enumerate(zip(indexes, selected)):
                replay_index = episode_index * 10 + local_index
                row = observations[replay_index]
                state = np.asarray(source_row["state"], dtype=np.float32)
                physical = hashlib.sha256()
                expected_images = {}
                expected_image_hashes = {}
                for camera in cameras:
                    encoded = bytes(source_row[camera]["bytes"])
                    name = (
                        f"episode_{episode_index:06d}_frame_{frame_index:06d}_{camera}.png"
                    )
                    relative_image = f"fixture/images/{name}"
                    expected_images[camera] = relative_image
                    expected_image_hashes[camera] = hashlib.sha256(encoded).hexdigest()
                    extracted = fixture_root / relative_image
                    if not extracted.is_file() or extracted.read_bytes() != encoded:
                        issue(issues, f"replay: Xiaomi extracted image differs at {replay_index} {camera}")
                        return
                    physical.update(encoded)
                physical.update(np.ascontiguousarray(state).tobytes())
                physical_hash = physical.hexdigest()
                expected_split = (
                    "diagnostic" if episode_index in diagnostic_episodes else "heldout"
                )
                expected_id = (
                    f"xiaomi-robocasa-episode-{episode_index:06d}-frame-{frame_index:06d}"
                )
                if (
                    row.get("id") != expected_id
                    or row.get("replay_index") != replay_index
                    or row.get("task_index") != source_row["task_index"]
                    or row.get("task") != tasks[source_row["task_index"]]
                    or row.get("episode_id") != f"episode_{episode_index:06d}"
                    or row.get("observation_index") != local_index
                    or row.get("source_frame_index") != frame_index
                    or row.get("split") != expected_split
                    or not np.array_equal(np.asarray(row.get("state"), dtype=np.float32), state)
                    or row.get("state_sha256")
                    != hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
                    or row.get("images") != expected_images
                    or row.get("image_sha256") != expected_image_hashes
                    or row.get("physical_observation_sha256") != physical_hash
                ):
                    issue(issues, f"replay: Xiaomi frozen source row differs at {replay_index}")
                    return
                physical_hashes.add(physical_hash)
        if len(physical_hashes) != 1000:
            issue(issues, f"replay: Xiaomi has only {len(physical_hashes)}/1000 unique observations")
    except (KeyError, OSError, TypeError, ValueError, ModuleNotFoundError) as error:
        issue(issues, f"replay: Xiaomi source could not be independently inspected: {error}")


def _vitra_dependencies():
    try:
        import av
        import h5py
        from PIL import Image
        from scipy.spatial.transform import Rotation
    except ModuleNotFoundError:
        dependency = Path("/workspace/envs/dexvla/lib/python3.12/site-packages")
        if not dependency.is_dir():
            raise
        sys.path.insert(0, str(dependency))
        import av
        import h5py
        from PIL import Image
        from scipy.spatial.transform import Rotation
    return av, h5py, Image, Rotation


def vitra_source_audit(
    manifest: dict, observations: list[dict], issues: list[str]
) -> None:
    """Independently verify VITRA's 50-episode/1,000-real-frame freeze."""
    expected_checkpoint = {
        "repository": "microsoft/VITRA-VLA-3B",
        "revision": VITRA_CHECKPOINT_REVISION,
        "filename": "vitra-vla-3b.pt",
        "bytes": VITRA_CHECKPOINT_BYTES,
        "sha256": VITRA_CHECKPOINT_SHA256,
    }
    if (
        manifest.get("dataset_revision") != VITRA_DATASET_REVISION
        or manifest.get("official_repository_revision") != VITRA_REPOSITORY_REVISION
        or manifest.get("checkpoint_revision") != VITRA_CHECKPOINT_REVISION
        or manifest.get("checkpoint_expected") != expected_checkpoint
        or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
    ):
        issue(issues, "replay: VITRA dataset/repository/checkpoint provenance differs")

    source_record = manifest.get("source") or {}
    source_root = Path(source_record.get("root", ""))
    recorded_sources = source_record.get("files") or []
    actual_sources = []
    for folder, extension in (("annotation", "h5"), ("videos", "mp4")):
        for episode in range(50):
            relative = f"{folder}/{episode:06d}.{extension}"
            path = source_root / relative
            if not path.is_file():
                issue(issues, f"replay: VITRA source file is missing: {relative}")
                return
            actual_sources.append({
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    actual_sources.sort(key=lambda row: row["path"])
    source_aggregate = hashlib.sha256(
        json.dumps(actual_sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        recorded_sources != actual_sources
        or sum(row["bytes"] for row in actual_sources) != VITRA_SOURCE_BYTES
        or source_aggregate != VITRA_SOURCE_AGGREGATE_SHA256
        or source_record.get("file_count") != 100
        or source_record.get("episodes") != 50
        or source_record.get("aggregate_sha256") != VITRA_SOURCE_AGGREGATE_SHA256
    ):
        issue(issues, "replay: VITRA pinned source inventory differs")
        return

    repository = Path((manifest.get("official_repository") or {}).get("path", ""))
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    if revision != VITRA_REPOSITORY_REVISION:
        issue(issues, "replay: VITRA official repository revision changed or disappeared")

    checkpoint = Path(manifest.get("checkpoint_root", "")) / "vitra-vla-3b.pt"
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != VITRA_CHECKPOINT_BYTES
        or sha256(checkpoint) != VITRA_CHECKPOINT_SHA256
    ):
        issue(issues, "replay: VITRA exact checkpoint is missing or changed")

    fixture_root = Path(manifest.get("fixture_root", ""))
    try:
        av, h5py, Image, Rotation = _vitra_dependencies()
        physical_hashes = set()
        replay_index = 0
        for episode_index in range(50):
            annotation_path = source_root / f"annotation/{episode_index:06d}.h5"
            video_path = source_root / f"videos/{episode_index:06d}.mp4"
            with h5py.File(annotation_path, "r") as episode:
                frame_count = int(episode["meta/frame_count"][()])
                instruction = episode["meta/instruction"][()].decode("utf-8")
                selected = [
                    int(value)
                    for value in np.linspace(0, frame_count - 1, 20, dtype=np.int64)
                ]
                decoded = {}
                wanted = set(selected)
                with av.open(str(video_path)) as container:
                    for frame_index, frame in enumerate(container.decode(video=0)):
                        if frame_index in wanted:
                            decoded[frame_index] = frame.to_image().convert("RGB")
                if set(decoded) != wanted:
                    issue(issues, f"replay: VITRA selected frames did not decode: {episode_index}")
                    return
                intrinsics = np.asarray(
                    episode["observation/camera/intrinsics"], dtype=np.float64
                )
                for local_index, frame_index in enumerate(selected):
                    row = observations[replay_index]
                    source_rgb = np.asarray(decoded[frame_index], dtype=np.uint8)
                    source_hash = hashlib.sha256(
                        np.ascontiguousarray(source_rgb).tobytes()
                    ).hexdigest()
                    width, height = decoded[frame_index].size
                    resized = decoded[frame_index].resize(
                        (int(width * 224 / height), 224), Image.Resampling.LANCZOS
                    )
                    resized_rgb = np.asarray(resized, dtype=np.uint8)
                    image_path = fixture_root / row.get("image", "")
                    if not image_path.is_file():
                        issue(issues, f"replay: VITRA fixture image is missing: {replay_index}")
                        return
                    fixture_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

                    pose = np.asarray(
                        episode["state/right_hand_mount_pose_in_cam"][frame_index],
                        dtype=np.float64,
                    )
                    joints = np.asarray(
                        episode["state/right_hand_joint"][frame_index], dtype=np.float64
                    )
                    pose_euler = pose.copy()
                    pose_euler[3:6] = Rotation.from_rotvec(pose[3:6]).as_euler(
                        "xyz", degrees=False
                    )
                    xhand_state = np.concatenate([pose_euler, joints]).astype(np.float32)
                    physical = hashlib.sha256()
                    physical.update(np.ascontiguousarray(source_rgb).tobytes())
                    physical.update(np.ascontiguousarray(pose).tobytes())
                    physical.update(np.ascontiguousarray(joints).tobytes())
                    physical_hash = physical.hexdigest()
                    expected_task = (
                        "pick_mango" if instruction.startswith("Pick up") else "place_mango"
                    )
                    expected_split = (
                        "diagnostic"
                        if episode_index in VITRA_DIAGNOSTIC_EPISODES
                        else "heldout"
                    )
                    expected_id = (
                        f"vitra-teledata-episode-{episode_index:06d}-"
                        f"frame-{frame_index:06d}"
                    )
                    if (
                        row.get("id") != expected_id
                        or row.get("replay_index") != replay_index
                        or row.get("episode_index") != episode_index
                        or row.get("episode_id") != f"episode_{episode_index:06d}"
                        or row.get("observation_index") != local_index
                        or row.get("source_frame_index") != frame_index
                        or row.get("split") != expected_split
                        or row.get("instruction") != instruction
                        or row.get("task") != expected_task
                        or row.get("prompt") != f"Left hand: None. Right hand: {instruction}"
                        or row.get("source_image_shape_hwc") != list(source_rgb.shape)
                        or row.get("source_image_rgb_sha256") != source_hash
                        or row.get("model_image_shape_hwc") != list(resized_rgb.shape)
                        or row.get("model_image_rgb_sha256")
                        != hashlib.sha256(np.ascontiguousarray(resized_rgb).tobytes()).hexdigest()
                        or row.get("model_image_png_sha256") != sha256(image_path)
                        or not np.array_equal(fixture_rgb, resized_rgb)
                        or not np.array_equal(
                            np.asarray(row.get("camera_intrinsics"), dtype=np.float64),
                            intrinsics,
                        )
                        or not np.array_equal(
                            np.asarray(row.get("source_pose_axis_angle"), dtype=np.float64),
                            pose,
                        )
                        or not np.array_equal(
                            np.asarray(row.get("source_joints"), dtype=np.float64), joints
                        )
                        or not np.array_equal(
                            np.asarray(row.get("xhand_state"), dtype=np.float32), xhand_state
                        )
                        or row.get("xhand_state_sha256")
                        != hashlib.sha256(np.ascontiguousarray(xhand_state).tobytes()).hexdigest()
                        or row.get("physical_observation_sha256") != physical_hash
                    ):
                        issue(issues, f"replay: VITRA frozen source row differs at {replay_index}")
                        return
                    physical_hashes.add(physical_hash)
                    replay_index += 1
        if replay_index != 1000 or len(physical_hashes) != 1000:
            issue(issues, f"replay: VITRA has only {len(physical_hashes)}/1000 unique observations")
    except (KeyError, IndexError, OSError, TypeError, ValueError, ModuleNotFoundError) as error:
        issue(issues, f"replay: VITRA source could not be independently inspected: {error}")


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def rollout_json_paths(directory: Path) -> list[Path]:
    return sorted((*directory.glob("*.json"), *directory.glob("*.json.gz")))


def stream_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def pi_fast_flow_length(flow_steps: object) -> int | None:
    """Validate and return a complete, contiguous PI0-FAST token trace length."""
    if not isinstance(flow_steps, dict) or not flow_steps:
        return None
    prefix = "flow.step_"
    suffix = ".prediction"
    required_metrics = {
        "exact", "unequal_element_count", "max_absolute_error", "mean_absolute_error",
        "rmse", "relative_l2_error", "nonfinite_count",
    }
    indices = []
    for label, metrics in flow_steps.items():
        if not isinstance(label, str) or not label.startswith(prefix) or not label.endswith(suffix):
            return None
        encoded = label[len(prefix):-len(suffix)]
        if len(encoded) != 3 or not encoded.isdigit() or not isinstance(metrics, dict):
            return None
        if not required_metrics.issubset(metrics):
            return None
        indices.append(int(encoded))
    if len(indices) > 256 or sorted(indices) != list(range(len(indices))):
        return None
    return len(indices)


def spatialvla_flow_length(flow_steps: object) -> int | None:
    """Validate SpatialVLA's nonempty contiguous autoregressive score prefix."""
    if not isinstance(flow_steps, dict) or not flow_steps:
        return None
    prefix = "generation.step_"
    suffix = ".scores"
    required_metrics = {
        "exact", "unequal_element_count", "max_absolute_error", "mean_absolute_error",
        "rmse", "relative_l2_error", "nonfinite_count",
    }
    indices = []
    for label, metrics in flow_steps.items():
        if not isinstance(label, str) or not label.startswith(prefix) or not label.endswith(suffix):
            continue
        encoded = label[len(prefix):-len(suffix)]
        if len(encoded) != 3 or not encoded.isdigit() or not isinstance(metrics, dict):
            return None
        if not required_metrics.issubset(metrics):
            return None
        indices.append(int(encoded))
    if not indices or len(indices) > 256 or sorted(indices) != list(range(len(indices))):
        return None
    return len(indices)


def univla_flow_length(flow_steps: object) -> int | None:
    """Validate UniVLA's nonempty contiguous constrained-token score prefix."""
    if not isinstance(flow_steps, dict) or not flow_steps:
        return None
    prefix = "flow.step_"
    suffix = ".scores"
    required_metrics = {
        "exact", "unequal_element_count", "max_absolute_error", "mean_absolute_error",
        "rmse", "relative_l2_error", "nonfinite_count",
    }
    indices = []
    for label, metrics in flow_steps.items():
        if not isinstance(label, str) or not label.startswith(prefix) or not label.endswith(suffix):
            continue
        encoded = label[len(prefix):-len(suffix)]
        if len(encoded) != 3 or not encoded.isdigit() or not isinstance(metrics, dict):
            return None
        if not required_metrics.issubset(metrics):
            return None
        indices.append(int(encoded))
    if not indices or len(indices) > 80 or sorted(indices) != list(range(len(indices))):
        return None
    return len(indices)


def issue(issues: list[str], message: str) -> None:
    issues.append(message)


def timing_input_selection_audit(selection: object, expected_count: int, issues: list[str]) -> None:
    """Verify that explicitly selected timing inputs are frozen diagnostic requests."""
    if selection is None:
        return
    if not isinstance(selection, dict):
        issue(issues, "performance: malformed replay input selection")
        return
    path_value = selection.get("replay_manifest_path")
    path = Path(path_value) if isinstance(path_value, str) else None
    if path is None or not path.is_file() or sha256(path) != selection.get("replay_manifest_sha256"):
        issue(issues, "performance: replay input manifest is missing or hash-mismatched")
        return
    rows = read_json(path).get("observations", [])
    indices = selection.get("replay_indices")
    if (not isinstance(indices, list) or len(indices) != expected_count
            or any(not isinstance(index, int) or index < 0 or index >= len(rows) for index in indices)
            or len(set(indices)) != len(indices)):
        issue(issues, "performance: replay input indices are incomplete, duplicate, or invalid")
        return
    selected = [rows[index] for index in indices]
    if any(row.get("replay_index") != index or row.get("split") != "diagnostic" for index, row in zip(indices, selected)):
        issue(issues, "performance: timing inputs are not frozen diagnostic rows")
    def task_key(row):
        return row.get("task_index") if row.get("task_index") is not None else row.get("task")
    diagnostic = [row for row in rows if row.get("split") == "diagnostic"]
    source_tasks = {task_key(row) for row in diagnostic}
    chosen_tasks = {task_key(row) for row in selected}
    if len(chosen_tasks) < min(expected_count, len(source_tasks)):
        issue(issues, "performance: timing selection does not cover available distinct diagnostic tasks")
    source_episodes = {(task_key(row), row.get("episode_id")) for row in diagnostic}
    chosen_episodes = {(task_key(row), row.get("episode_id")) for row in selected}
    if len(chosen_episodes) < min(expected_count, len(source_episodes)):
        issue(issues, "performance: timing selection reuses an episode despite available distinct episodes")
    for field, expected in (
        ("episode_ids", [row.get("episode_id") for row in selected]),
        ("task_indices", [row.get("task_index") for row in selected]),
        ("request_ids", [row.get("request_ids", [None])[:1][0] if row.get("request_ids") else None for row in selected]),
    ):
        if selection.get(field) != expected:
            issue(issues, f"performance: timing input {field} differs from frozen replay")


def timing_audit(
    path: Path, kind: str, issues: list[str], *,
    required_policy_batches=(1, 2, 4, 8), expected_policy_modes=POLICY_MODES,
) -> dict:
    start = len(issues)
    if not path.is_file():
        issue(issues, f"{kind}: missing {path}")
        return {"complete": False, "timed": 0, "warmups": 0}
    environments = []
    fixed_shapes = []
    timed = defaultdict(set)
    warmups = defaultdict(lambda: defaultdict(set))
    seen_attempt_order = defaultdict(list)
    physical_sizes = defaultdict(set)
    logical_sizes = defaultdict(set)
    required_configuration = EXPECTED_EXECUTION_CONFIG.get(path.parents[2].name)
    for item in stream_jsonl(path):
        record_type = item.get("record_type")
        if record_type == "environment":
            environments.append(item)
            continue
        if record_type == "fixed_shape":
            if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
                issue(issues, "serving: fixed-padded shape recorded for another repaired path")
            fixed_shapes.append(item)
            continue
        if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
            issue(issues, f"{kind}: adapter execution configuration differs from repaired policy path")
            break
        if kind == "policy":
            key = (item.get("implementation"), item.get("batch_size"), item.get("session"))
        else:
            key = (item.get("configuration"), item.get("session"))
        if record_type == "timing":
            repetition = item.get("repetition")
            if repetition in timed[key]:
                issue(issues, f"{kind}: duplicate timing key {key}, repetition {repetition}")
            timed[key].add(repetition)
            seen_attempt_order[key].append("timing")
            if kind == "serving":
                physical_sizes[key].add(item.get("physical_batch_size"))
                logical_sizes[key].add(item.get("logical_batch_size"))
            if not isinstance(item.get("latency_seconds"), (float, int)) or item["latency_seconds"] <= 0:
                issue(issues, f"{kind}: invalid latency at {key}, repetition {repetition}")
            else:
                requested = item.get("batch_size") if kind == "policy" else item.get("logical_batch_size")
                throughput = item.get("requests_per_second")
                if not isinstance(requested, int) or requested < 1 or not isinstance(throughput, (int, float)) or abs(throughput - requested / item["latency_seconds"]) > max(1e-5, throughput * 1e-4):
                    issue(issues, f"{kind}: throughput arithmetic mismatch at {key}, repetition {repetition}")
                if not isinstance(item.get("peak_incremental_bytes"), int) or item["peak_incremental_bytes"] < 0:
                    issue(issues, f"{kind}: missing peak-memory evidence at {key}, repetition {repetition}")
                if kind == "serving":
                    expected_physical = (
                        1 if item.get("configuration") == "serial_singleton"
                        else item.get("logical_batch_size")
                        if item.get("configuration") in ("native_dynamic", "batch_invariant_dynamic")
                        else None
                    )
                    if expected_physical is not None and item.get("physical_batch_size") != expected_physical:
                        issue(issues, f"serving: physical batch-size mismatch at {key}")
        elif record_type == "warmup":
            attempt = item.get("attempt_id")
            index = item.get("warmup_index")
            if index in warmups[key][attempt]:
                issue(issues, f"{kind}: duplicate warmup {key}, attempt {attempt}, index {index}")
            warmups[key][attempt].add(index)
            seen_attempt_order[key].append(("warmup", attempt, index))
        else:
            issue(issues, f"{kind}: unexpected record type {record_type!r}")
    if len(environments) != 1:
        issue(issues, f"{kind}: expected one environment record, found {len(environments)}")
        return {"complete": False, "timed": sum(map(len, timed.values())), "warmups": 0}
    env = environments[0]
    if (env.get("sessions"), env.get("warmup"), env.get("repetitions")) != (SESSIONS, WARMUPS, TIMED_CALLS):
        issue(issues, f"{kind}: schedule metadata differs from hard-coded 5x20x200")
    if env.get("campaign_protocol") != "full_v1":
        issue(issues, f"{kind}: full protocol marker missing")
    if env.get("input_selection") is None:
        issue(issues, f"{kind}: missing frozen diagnostic input selection")
    if kind == "policy":
        if fixed_shapes:
            issue(issues, "policy: unexpected fixed-shape serving record")
        batches = env.get("batch_sizes", [])
        timing_input_selection_audit(env.get("input_selection"), max(required_policy_batches), issues)
        if set(batches) != set(required_policy_batches):
            issue(issues, f"policy: expected exactly batch sizes {required_policy_batches}, found {batches}")
        if set(env.get("implementations", POLICY_MODES)) != set(expected_policy_modes):
            issue(issues, f"policy: implementation modes differ from expected {expected_policy_modes}")
        expected = {(mode, batch, session) for mode in expected_policy_modes for batch in required_policy_batches for session in range(SESSIONS)}
    else:
        if len(fixed_shapes) != 1:
            issue(issues, f"serving: expected one fixed-padded input-shape record, found {len(fixed_shapes)}")
        else:
            shape = fixed_shapes[0].get("metadata") or {}
            if shape.get("physical_batch_size") != env.get("fixed_batch_size"):
                issue(issues, "serving: fixed-padded physical shape differs from environment")
            required_shape = EXPECTED_FIXED_LANGUAGE_SHAPE.get(path.parents[2].name)
            if required_shape is not None and any(shape.get(key) != value for key, value in required_shape.items()):
                issue(issues, "serving: fixed-padded prompt shape is not the audited canonical length")
        timing_input_selection_audit(env.get("input_selection"), env.get("fixed_batch_size", 0), issues)
        expected = {(mode, session) for mode in SERVING_MODES for session in range(SESSIONS)}
        for key in expected:
            if logical_sizes.get(key, set()) != {env.get("logical_batch_size")}:
                issue(issues, f"serving: logical batch shape varies or differs from environment at {key}")
            wanted = env.get("fixed_batch_size") if key[0] == "fixed_size_padded" else 1 if key[0] == "serial_singleton" else env.get("logical_batch_size")
            if physical_sizes.get(key, set()) != {wanted}:
                issue(issues, f"serving: physical batch shape varies or differs from environment at {key}")
    if set(timed) != expected:
        issue(issues, f"{kind}: expected {len(expected)} timing groups, found {len(timed)}")
    for key in expected:
        if timed.get(key, set()) != set(range(TIMED_CALLS)):
            issue(issues, f"{kind}: {key} has {len(timed.get(key, set()))}/{TIMED_CALLS} unique timed calls")
        attempts = warmups.get(key, {})
        if not attempts:
            issue(issues, f"{kind}: {key} has no recorded warmup attempt")
        if attempts and not any(indexes == set(range(WARMUPS)) for indexes in attempts.values()):
            issue(issues, f"{kind}: {key} has no complete 20-call warmup attempt")
        sequence = seen_attempt_order.get(key, [])
        current_attempt = None
        current_indexes = set()
        for event in sequence:
            if event == "timing":
                if current_indexes != set(range(WARMUPS)):
                    issue(issues, f"{kind}: {key} timed without 20 preceding warmups")
                    break
            else:
                _, attempt_id, index = event
                if attempt_id != current_attempt:
                    current_attempt = attempt_id
                    current_indexes = set()
                current_indexes.add(index)
    return {
        "complete": len(issues) == start,
        "timed": sum(map(len, timed.values())),
        "warmups": sum(len(indexes) for attempts in warmups.values() for indexes in attempts.values()),
        "groups": len(timed),
    }


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def uva_joint_regenerated_noise_audit(
    manifest: dict, observations: list[dict], issues: list[str], start: int,
) -> None:
    """Verify UVA-joint's compact, deterministic video-noise contract."""
    contract = manifest.get("noise_regeneration_contract") or {}
    expected_contract = {
        "schema_version": 1,
        "algorithm": "torch_cpu_generator_manual_seed_randn",
        "torch_version": "2.8.0+cu128",
        "device": "cpu", "dtype": "float32",
        "seed_base": 288000,
        "seed_formula": "seed_base + replay_index * 3 + noise_id",
        "draw_order": ["video_initial_noise", "video_step_noise"],
        "components": {
            "video_initial_noise": [1024, 16],
            "video_step_noise": [100, 1024, 16],
        },
    }
    if contract != expected_contract:
        issue(issues, "replay: UVA joint deterministic video-noise contract differs")
        return
    if manifest.get("noise_regeneration_contract_sha256") != canonical_sha256(contract):
        issue(issues, "replay: UVA joint video-noise contract hash differs")
        return
    stored_path = Path(manifest.get("stored_action_noise_file", ""))
    if (not stored_path.is_file()
            or sha256(stored_path) != manifest.get("stored_action_noise_file_sha256")
            or manifest.get("stored_action_noise_shape") != [1000, 3, 32544]):
        issue(issues, "replay: UVA joint stored action-noise tensor is missing or changed")
        return
    stored = np.load(stored_path, mmap_mode="r", allow_pickle=False)
    if list(stored.shape) != [1000, 3, 32544] or stored.dtype != np.float32:
        issue(issues, "replay: UVA joint stored action-noise tensor shape/dtype differs")
        return
    for row in observations:
        replay_index = row.get("replay_index")
        if not isinstance(replay_index, int) or not 0 <= replay_index < 1000:
            issue(issues, f"replay: UVA joint invalid regeneration index for {row.get('id')}")
            break
        noise_records = row.get("noise") or []
        if len(noise_records) != 3:
            issue(issues, f"replay: UVA joint lacks three noise contracts for {row.get('id')}")
            break
        for noise_id, record in enumerate(noise_records):
            stored_hash = hashlib.sha256(
                np.asarray(stored[replay_index, noise_id]).tobytes()
            ).hexdigest()
            video_contract = {
                "contract_sha256": manifest["noise_regeneration_contract_sha256"],
                "replay_index": replay_index, "noise_id": noise_id,
                "seed": 288000 + replay_index * 3 + noise_id,
            }
            video_hash = canonical_sha256(video_contract)
            request_hash = canonical_sha256({
                "stored_action_sha256": stored_hash,
                "video_contract_sha256": video_hash,
            })
            if record != {
                "noise_id": noise_id,
                "stored_action_sha256": stored_hash,
                "video_seed": video_contract["seed"],
                "video_contract_sha256": video_hash,
                "sha256": request_hash,
            }:
                issue(issues, f"replay: UVA joint noise contract mismatch for {row.get('id')} #{noise_id}")
                break
        if len(issues) > start:
            break


def openhelix_source_audit(
    manifest: dict, observations: list[dict], issues: list[str]
) -> None:
    """Independently inspect the 1,000 retained pinned CALVIN transitions."""
    dataset = manifest.get("dataset") or {}
    checkpoint = manifest.get("checkpoint") or {}
    expected_checkpoint = [
        {"path": name, "bytes": size, "sha256": digest}
        for name, size, digest in OPENHELIX_CHECKPOINT_FILES
    ]
    if (
        dataset.get("repository") != "KyujinL/CALVIN_ABC"
        or dataset.get("revision") != OPENHELIX_DATASET_REVISION
        or dataset.get("archive_filename") != "validation.tar.part-000"
        or dataset.get("archive_bytes") != OPENHELIX_ARCHIVE_BYTES
        or dataset.get("archive_sha256") != OPENHELIX_ARCHIVE_SHA256
        or dataset.get("selection") != "first_1000_regular_transition_members_in_tar_order"
        or dataset.get("streamed_prefix_only") is not True
        or manifest.get("official_repository_revision") != OPENHELIX_REPOSITORY_REVISION
        or checkpoint.get("repository") != "OpenHelix/openhelix"
        or checkpoint.get("revision") != OPENHELIX_CHECKPOINT_REVISION
        or checkpoint.get("files") != expected_checkpoint
    ):
        issue(issues, "replay: OpenHelix pinned source/repository/checkpoint provenance differs")

    fixture_root = Path(manifest.get("fixture_root", ""))
    source_files = manifest.get("source_files") or []
    if (
        len(source_files) != 1000
        or manifest.get("source_files_canonical_sha256") != canonical_sha256(source_files)
        or manifest.get("unique_physical_observations") != 1000
    ):
        issue(issues, "replay: OpenHelix retained source inventory is incomplete or changed")
        return

    expected_arrays = {
        "rgb_static": ((200, 200, 3), np.dtype("uint8")),
        "rgb_gripper": ((84, 84, 3), np.dtype("uint8")),
        "depth_static": ((200, 200), np.dtype("float32")),
        "depth_gripper": ((84, 84), np.dtype("float32")),
        "robot_obs": ((15,), np.dtype("float64")),
    }
    physical_hashes = set()
    for index, (record, row) in enumerate(zip(source_files, observations)):
        expected_name = row.get("source_file")
        path = fixture_root / str(expected_name)
        if (
            record.get("path") != f"validation/{expected_name}"
            or record.get("bytes") != row.get("source_bytes")
            or record.get("sha256") != row.get("source_sha256")
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            issue(issues, f"replay: OpenHelix source file changed at index {index}")
            break
        array_hashes = {}
        physical = hashlib.sha256()
        try:
            with np.load(path, allow_pickle=False) as item:
                for name, (shape, dtype) in expected_arrays.items():
                    value = item[name]
                    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                        raise ValueError(f"invalid {name} shape/dtype/value")
                    contiguous = np.ascontiguousarray(value)
                    array_hashes[name] = hashlib.sha256(contiguous.tobytes()).hexdigest()
                    physical.update(name.encode() + b"\0")
                    physical.update(contiguous.tobytes())
        except (KeyError, OSError, TypeError, ValueError) as error:
            issue(issues, f"replay: OpenHelix source cannot be decoded at index {index}: {error}")
            break
        physical_hash = physical.hexdigest()
        expected_companions = [(index + offset) % 1000 for offset in range(1, 32)]
        if (
            row.get("id") != f"openhelix-calvin-{Path(str(expected_name)).stem}"
            or row.get("replay_index") != index
            or row.get("episode_id") != Path(str(expected_name)).stem
            or row.get("observation_index") != 0
            or row.get("source_member") != f"validation/{expected_name}"
            or row.get("consumed_array_sha256") != array_hashes
            or row.get("image_sha256") != {
                "static": array_hashes["rgb_static"],
                "gripper": array_hashes["rgb_gripper"],
            }
            or row.get("physical_observation_sha256") != physical_hash
            or row.get("instruction") != "push the sliding door to the right side"
            or row.get("task") != "open_drawer"
            or row.get("split") != ("diagnostic" if index < 200 else "heldout")
            or row.get("companion_replay_indices") != expected_companions
        ):
            issue(issues, f"replay: OpenHelix frozen observation differs at index {index}")
            break
        physical_hashes.add(physical_hash)
    if len(physical_hashes) != 1000:
        issue(issues, f"replay: OpenHelix has only {len(physical_hashes)}/1000 verified unique physical observations")


def replay_audit(directory: Path, issues: list[str]) -> dict:
    start = len(issues)
    model_id = directory.parent.name
    expected_observations, expected_diagnostic, expected_heldout = expected_replay_counts(model_id)
    manifest = read_json(directory / "replay_manifest.json")
    freeze = read_json(directory / "numerical_freeze.json")
    if not manifest:
        issue(issues, "replay: missing frozen replay_manifest.json")
    if not freeze:
        issue(issues, "replay: missing numerical_freeze.json")
    else:
        frozen_files = freeze.get("content_addressed_freeze")
        if not frozen_files or not freeze.get("invariant_diagnostic_exact"):
            issue(issues, "replay: numerical freeze lacks content-addressed exact diagnostic evidence")
        else:
            # The freeze writer validates and hashes evidence but never enters
            # model execution.  Older freezes retained its own hash; changing
            # that provenance-only producer must not invalidate already-frozen
            # numerical outputs.  All model, harness, kernel, replay,
            # checkpoint, capacity, and diagnostic bytes remain fail-closed.
            freeze_producer = Path(__file__).resolve().with_name("freeze_model_numerics.py")
            for filename, expected in frozen_files.items():
                path = Path(filename)
                if path.resolve() == freeze_producer:
                    continue
                if not path.is_file() or path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get("sha256"):
                    issue(issues, f"replay: frozen source/checkpoint changed or missing: {filename}")
                    break
    if not manifest:
        return {"complete": False}
    if manifest.get("model_id") and manifest["model_id"] != model_id:
        issue(issues, "replay: frozen manifest belongs to another model")
    observations = manifest.get("observations", [])
    if manifest.get("source_replay_manifest_sha256"):
        source_manifest_path = Path(manifest.get("source_replay_manifest_path", ""))
        if not source_manifest_path.is_file() or sha256(source_manifest_path) != manifest["source_replay_manifest_sha256"]:
            issue(issues, "replay: derived real-input source manifest hash changed or disappeared")
        else:
            source_observations = (read_json(source_manifest_path) or {}).get("observations", [])
            source_fields = ("id", "replay_index", "task", "episode_id", "observation_index", "split", "state", "images", "image_sha256", "companion_replay_indices")
            if len(source_observations) != len(observations) or any(
                any(source.get(field) != derived.get(field) for field in source_fields)
                for source, derived in zip(source_observations, observations)
            ):
                issue(issues, "replay: derived manifest altered source observation membership or pixels")
    ids = [item.get("id") for item in observations]
    if len(ids) != expected_observations or len(set(ids)) != expected_observations:
        issue(issues, f"replay: expected {expected_observations} unique observations, found {len(ids)} rows / {len(set(ids))} IDs")
    indexes = [item.get("replay_index") for item in observations]
    if (len(indexes) != expected_observations
            or not all(isinstance(index, int) for index in indexes)
            or set(indexes) != set(range(expected_observations))):
        issue(issues, f"replay: observation-to-noise indices are not a one-to-one 0..{expected_observations - 1} mapping")
    schedule = manifest.get("companion_schedule") or {}
    expected_schedule = (
        "distinct_source_observations_v1"
        if model_id in {"dexvla", "lingbot_vla2", "openhelix"}
        else "unrelated_episode_heads_v1"
    )
    if schedule.get("version") != expected_schedule or schedule.get("max_companions", 0) < 31:
        issue(issues, "replay: missing frozen B32-capable unrelated-companion schedule")
    elif model_id in {"dexvla", "lingbot_vla2", "openhelix"}:
        for row in observations:
            companion_indexes = row.get("companion_replay_indices", [])
            if (len(companion_indexes) != 31 or len(set(companion_indexes)) != 31
                    or row.get("replay_index") in companion_indexes
                    or any(not isinstance(index, int) or not 0 <= index < len(observations)
                           for index in companion_indexes)):
                issue(issues, f"replay: invalid source-limited companion schedule for {row.get('id')}")
                break
    elif len(observations) == expected_observations:
        episode_heads = {}
        for row in observations:
            episode_heads.setdefault((row.get("task"), row.get("episode_id")), row.get("replay_index"))
        for row in observations:
            companion_indexes = row.get("companion_replay_indices", [])
            if len(companion_indexes) != 31 or len(set(companion_indexes)) != 31 or any(
                not isinstance(index, int) or not 0 <= index < len(observations)
                for index in companion_indexes
            ):
                issue(issues, f"replay: invalid companion schedule for {row.get('id')}")
                break
            companions = [observations[index] for index in companion_indexes]
            episodes = [(item.get("task"), item.get("episode_id")) for item in companions]
            other_tasks = {item.get("task") for item in observations if item.get("task") != row.get("task")}
            distinct_other_tasks_required = min(7, len(other_tasks))
            other_task_episode_count = sum(task != row.get("task") for task, _ in episode_heads)
            if (
                len(set(episodes)) != 31
                or (row.get("task"), row.get("episode_id")) in episodes
                or any(episode_heads.get(episode) != index for episode, index in zip(episodes, companion_indexes))
                or len({item.get("task") for item in companions[:distinct_other_tasks_required]}) != distinct_other_tasks_required
                or any(item.get("task") == row.get("task") for item in companions[:distinct_other_tasks_required])
                or (other_task_episode_count >= 31 and any(item.get("task") == row.get("task") for item in companions))
            ):
                issue(issues, f"replay: companions are not distinct unrelated episode heads for {row.get('id')}")
                break
    splits = Counter(item.get("split") for item in observations)
    expected_splits = {"diagnostic": expected_diagnostic, "heldout": expected_heldout}
    if splits != expected_splits:
        issue(issues, f"replay: split counts are {dict(splits)}, expected {expected_diagnostic}/{expected_heldout}")
    multiplicity = request_multiplicity(manifest, issues)
    episode_rows = Counter(item.get("episode_id") for item in observations)
    diagnostic_episodes = {item.get("episode_id") for item in observations if item.get("split") == "diagnostic"}
    heldout_episodes = {item.get("episode_id") for item in observations if item.get("split") == "heldout"}
    if model_id in {"dexvla", "lingbot_vla2"}:
        if episode_rows != {"episode_47": 396}:
            issue(issues, f"replay: {model_id} must use all 396 frames of its only unique public episode")
        if diagnostic_episodes != {"episode_47"} or heldout_episodes != {"episode_47"}:
            issue(issues, f"replay: {model_id} source-limited frame split does not cover the sole episode")
    elif model_id == "rdt_1b":
        if len(episode_rows) != 250 or set(episode_rows.values()) != {4}:
            issue(issues, "replay: RDT requires 250 distinct episodes with four observations each")
        if (len(diagnostic_episodes) != 50 or len(heldout_episodes) != 200
                or diagnostic_episodes & heldout_episodes):
            issue(issues, "replay: RDT episode-level 50/200 diagnostic/heldout split is not isolated")
    elif model_id == "vitra_vla_3b":
        if len(episode_rows) != 50 or set(episode_rows.values()) != {20}:
            issue(issues, "replay: VITRA requires 50 distinct episodes with 20 observations each")
        if (len(diagnostic_episodes) != 10 or len(heldout_episodes) != 40
                or diagnostic_episodes & heldout_episodes):
            issue(issues, "replay: VITRA episode-level 10/40 diagnostic/heldout split is not isolated")
    elif model_id == "openhelix":
        if len(episode_rows) != 1000 or set(episode_rows.values()) != {1}:
            issue(issues, "replay: OpenHelix requires 1,000 distinct public CALVIN transitions")
        if (len(diagnostic_episodes) != 200 or len(heldout_episodes) != 800
                or diagnostic_episodes & heldout_episodes):
            issue(issues, "replay: OpenHelix 200/800 transition split is not isolated")
    else:
        if len(episode_rows) != 100 or set(episode_rows.values()) != {10}:
            issue(issues, "replay: require 100 distinct episodes with 10 observations each")
        if (len(diagnostic_episodes) != 20 or len(heldout_episodes) != 80
                or diagnostic_episodes & heldout_episodes):
            issue(issues, "replay: episode-level 20/80 diagnostic/heldout split is not isolated")
    diagnostic_tasks = {item.get("task") for item in observations if item.get("split") == "diagnostic"}
    heldout_tasks = {item.get("task") for item in observations if item.get("split") == "heldout"}
    if diagnostic_tasks != heldout_tasks:
        issue(issues, "replay: diagnostic and held-out task sets differ")
    request_ids = [request_id for item in observations for request_id in item.get("request_ids", [])]
    expected_request_count = expected_observations * multiplicity
    if len(request_ids) != expected_request_count or len(set(request_ids)) != expected_request_count:
        issue(issues, f"replay: require {expected_request_count} distinct observation/request IDs")
    fixture_root = manifest.get("fixture_root")
    noise_file = manifest.get("noise_file")
    if multiplicity == 3 and fixture_root and noise_file:
        noise_path = Path(fixture_root) / noise_file
        if not noise_path.is_file() or sha256(noise_path) != manifest.get("noise_file_sha256"):
            issue(issues, "replay: frozen noise file missing or hash mismatch")
        else:
            noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)
            if (list(noise.shape) != manifest.get("noise_shape")
                    or noise.shape[:2] != (expected_observations, 3)):
                issue(issues, "replay: explicit noise tensor shape differs from manifest")
            else:
                for item in observations:
                    index = item.get("replay_index")
                    if not isinstance(index, int) or not 0 <= index < expected_observations:
                        issue(issues, f"replay: invalid noise index for {item.get('id')}")
                        break
                    for noise_id, record in enumerate(item.get("noise", [])):
                        if record.get("noise_id") != noise_id or hashlib.sha256(np.asarray(noise[index, noise_id]).tobytes()).hexdigest() != record.get("sha256"):
                            issue(issues, f"replay: stored noise hash mismatch for {item.get('id')} #{noise_id}")
                            break
    elif (multiplicity == 3 and directory.parent.name == "uva_joint"
          and manifest.get("noise_regeneration_contract")):
        uva_joint_regenerated_noise_audit(manifest, observations, issues, start)
    elif multiplicity == 3:
        issue(issues, "replay: explicit noise storage is not auditable")
    if fixture_root and model_id in {"dexvla", "lingbot_vla2"}:
        source_limited_hdf5_audit(model_id, manifest, observations, issues)
    elif fixture_root and model_id == "xiaomi_robotics_1":
        xiaomi_source_audit(manifest, observations, issues)
    elif fixture_root and model_id == "vitra_vla_3b":
        vitra_source_audit(manifest, observations, issues)
    elif fixture_root and model_id == "openhelix":
        openhelix_source_audit(manifest, observations, issues)
    elif fixture_root:
        for item in observations:
            for camera, relative_path in item.get("images", {}).items():
                expected_hash = item.get("image_sha256", {}).get(camera)
                if relative_path is None and directory.parent.name == "rdt_1b":
                    history_index = int(camera.rsplit("_", 1)[-1])
                    valid = item.get("history_valid", [])
                    if (expected_hash is not None or len(valid) != 2
                            or valid[history_index] is not False):
                        issue(issues, f"replay: malformed absent RDT history image {item.get('id')} {camera}")
                    continue
                path = Path(fixture_root) / relative_path
                if not path.is_file() or not expected_hash or sha256(path) != expected_hash:
                    issue(issues, f"replay: missing or altered image {item.get('id')} {camera}")
                    break
            if len(issues) > start + 20:
                break
        source_root = manifest.get("source_root")
        if not manifest.get("source_sha256"):
            issue(issues, "replay: no hashed real-input source files")
        for relative_path, expected_hash in manifest.get("source_sha256", {}).items():
            path = Path(source_root) / relative_path if source_root else Path(relative_path)
            if not path.is_file() or sha256(path) != expected_hash:
                issue(issues, f"replay: missing or altered source parquet {relative_path}")
                break
    else:
        issue(issues, "replay: fixture root is missing")
    if model_id in ("uva", "uva_joint"):
        uva_context_audit(manifest, observations, issues, joint=model_id == "uva_joint")
    return {"complete": len(issues) == start, "observations": len(ids), "episodes": len(episode_rows), "splits": dict(splits), "requests_per_observation": multiplicity}


def uva_context_audit(manifest: dict, observations: list[dict], issues: list[str], *, joint: bool) -> None:
    """Independently verify UVA's actual 16-frame windows and model-shaped noises."""
    import pyarrow.parquet as pq

    if (manifest.get("checkpoint_sha256") != UVA_LIBERO10_CHECKPOINT_SHA256
            or not Path(manifest.get("checkpoint_path", "")).is_file()
            or sha256(Path(manifest["checkpoint_path"])) != UVA_LIBERO10_CHECKPOINT_SHA256):
        issue(issues, "replay: UVA does not use the exact pinned LIBERO-10 checkpoint")
    layouts = {
        "vae_noise": (0, [4, 16, 16, 16]),
        "action_initial_noise": (16384, [16, 10]),
        "action_step_noise": (16544, [100, 16, 10]),
    }
    if joint:
        layouts.update({
            "video_initial_noise": (32544, [1024, 16]),
            "video_step_noise": (48928, [100, 1024, 16]),
        })
    width = 1687328 if joint else 32544
    actual_layout = manifest.get("noise_layout") or {}
    if (manifest.get("noise_shape") != [1000, 3, width]
            or set(actual_layout) != set(layouts)
            or any(actual_layout.get(name) != {"offset": offset, "shape": shape}
                   for name, (offset, shape) in layouts.items())):
        issue(issues, "replay: UVA noise storage lacks the exact VAE/action/video tensors")
    inventory_path = Path(manifest.get("context_inventory_path", ""))
    if (not inventory_path.is_file()
            or sha256(inventory_path) != manifest.get("context_inventory_sha256")):
        issue(issues, "replay: UVA source-window inventory is missing or changed")
        return
    inventory = read_json(inventory_path) or {}
    windows = inventory.get("windows", [])
    if (inventory.get("source_replay_manifest_sha256") != manifest.get("source_replay_manifest_sha256")
            or inventory.get("source_root") != manifest.get("source_root")
            or inventory.get("distinct_16_frame_contexts") != 1000
            or len(windows) != 1000 or len(observations) != 1000
            or any(row.get("context_window") != window for row, window in zip(observations, windows))
            or len({(window.get("episode_id"), window.get("start_frame_index")) for window in windows}) != 1000):
        issue(issues, "replay: UVA contexts are not 1,000 distinct pinned real 16-frame windows")
        return
    expected_files = inventory.get("source_parquet_sha256") or {}
    if not expected_files or any(manifest.get("source_sha256", {}).get(name) != digest
                                 for name, digest in expected_files.items()):
        issue(issues, "replay: UVA window source hashes differ from the frozen parquet corpus")
        return
    metadata = {}
    try:
        for name in expected_files:
            path = Path(inventory["source_root"]) / name
            metadata[name] = pq.read_table(
                path, columns=["episode_index", "frame_index", "task_index"]
            ).to_pylist()
        for row, window in zip(observations, windows):
            source_rows = metadata[window["source_parquet"]]
            start = window["source_row_start"]
            if (not isinstance(start, int) or start < 0 or start + 16 > len(source_rows)
                    or window["end_frame_index"] - window["start_frame_index"] != 15
                    or not window["start_frame_index"] <= window["anchor_frame_index"] <= window["end_frame_index"]
                    or window["anchor_frame_index"] != row["observation_index"]):
                issue(issues, f"replay: UVA window bounds differ for {row.get('id')}")
                break
            if any(source_rows[start + offset] != {
                "episode_index": row["episode_id"],
                "frame_index": window["start_frame_index"] + offset,
                "task_index": row["task_index"],
            } for offset in range(16)):
                issue(issues, f"replay: UVA context is not 16 real contiguous frames for {row.get('id')}")
                break
    except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
        issue(issues, f"replay: UVA context source could not be verified: {error}")


def diagnostic_audit(directory: Path, issues: list[str], capacity: dict | None = None) -> dict:
    """Independently recount every frozen diagnostic request and arrangement."""
    start = len(issues)
    raw = directory / "diagnostic" / "invariance.jsonl.gz"
    fidelity = directory / "diagnostic" / "singleton_fidelity.jsonl.gz"
    if not raw.is_file() or not fidelity.is_file():
        issue(issues, "diagnostic: missing raw comparison or singleton-fidelity archive")
        return {"complete": False}
    manifest_path = directory / "replay_manifest.json"
    manifest = read_json(manifest_path) or {}
    if not manifest_path.is_file():
        issue(issues, "diagnostic: missing replay manifest")
        return {"complete": False}
    manifest_hash = sha256(manifest_path)
    multiplicity = request_multiplicity(manifest, issues)
    sources = {
        request_id: (row, noise_id)
        for row in manifest.get("observations", []) if row.get("split") == "diagnostic"
        for noise_id, request_id in enumerate(row.get("request_ids", []))
    }
    expected_requests = 200 * multiplicity
    if len(sources) != expected_requests:
        issue(issues, f"diagnostic: expected {expected_requests} unique real request/noise pairs")
    capacity = capacity or {}
    required = {
        mode: required_arrangements(replay_batch_sizes(directory.parent.name, capacity, mode))
        for mode in POLICY_MODES
    }
    required_configuration = EXPECTED_EXECUTION_CONFIG.get(directory.parent.name)
    groups = defaultdict(set)
    total = 0
    required_metric_fields = {
        "exact", "unequal_element_count", "max_absolute_error", "mean_absolute_error",
        "rmse", "relative_l2_error", "nonfinite_count",
    }
    for item in stream_jsonl(raw):
        total += 1
        request_id = item.get("request_id")
        mode = item.get("implementation")
        source = sources.get(request_id)
        if source is None or mode not in POLICY_MODES:
            issue(issues, f"diagnostic: unknown request or mode {request_id}, {mode}")
            break
        row, noise_id = source
        if item.get("replay_manifest_sha256") != manifest_hash or item.get("numerical_freeze_sha256") is not None:
            issue(issues, "diagnostic: raw record is not bound to the selected unfrozen replay")
            break
        if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
            issue(issues, f"diagnostic: unrepaired or unstamped execution configuration for {directory.parent.name}")
            break
        if (item.get("task"), item.get("episode_id"), item.get("observation_index"), item.get("noise_id")) != (
            row.get("task"), row.get("episode_id"), row.get("observation_index"), noise_id
        ):
            issue(issues, f"diagnostic: request metadata differs from frozen source for {request_id}")
            break
        expected_noise_hash = row.get("noise", [])[noise_id]["sha256"] if multiplicity == 3 else None
        if item.get("source_image_sha256") != observation_image_sha256(row) or item.get("request_noise_sha256") != expected_noise_hash:
            issue(issues, f"diagnostic: real-input/noise hash mismatch for {request_id}")
            break
        logical = item.get("logical_batch_size")
        physical = item.get("batch_size")
        target_index = item.get("target_batch_index")
        if not isinstance(logical, int) or logical < 1 or not isinstance(target_index, int) or not 0 <= target_index < logical:
            issue(issues, f"diagnostic: invalid logical size or target index for {request_id}")
            break
        arrangement = (
            item.get("transformation"), logical,
            item.get("target_batch_position"), item.get("companion_type"),
        )
        if arrangement not in required[mode] or arrangement in groups[(request_id, mode)]:
            issue(issues, f"diagnostic: duplicated or unplanned arrangement for {request_id}, {mode}: {arrangement}")
            break
        groups[(request_id, mode)].add(arrangement)
        companions = [
            manifest["observations"][index]["request_ids"][noise_id]
            for index in row.get("companion_replay_indices", [])[: logical - 1]
        ]
        if item.get("companion_type") == "duplicate":
            companions = [request_id] * (logical - 1)
        if item.get("transformation") == "singleton":
            expected_order = [request_id]
        elif item.get("transformation") == "batch_position":
            expected_order = list(companions)
            expected_order.insert(target_index, request_id)
        elif item.get("transformation") == "permutation":
            expected_order = list(reversed([request_id, *companions]))
        else:
            expected_order = [request_id, *companions]
        if item.get("request_ordering") != expected_order or item.get("companion_ids") != [
            identifier for index, identifier in enumerate(expected_order) if index != target_index
        ]:
            issue(issues, f"diagnostic: companion ordering differs from frozen schedule for {request_id}")
            break
        expected_physical = max(1, logical // 2) if item.get("transformation") == "partition" else logical
        if physical != expected_physical:
            issue(issues, f"diagnostic: physical batch size differs from arrangement for {request_id}")
            break
        output = item.get("output") or {}
        if not required_metric_fields.issubset(output) or not isinstance(item.get("flow_steps"), dict):
            issue(issues, f"diagnostic: missing output metrics or iterative trace for {request_id}")
            break
        if directory.parent.name == "pi_fast" and not {
            "output.actions", "output.official_decode_valid",
        }.issubset((output.get("tensors") or {}).keys()):
            issue(issues, f"diagnostic: PI0-FAST omits official decode validity for {request_id}")
            break
        if directory.parent.name == "pi_fast" and pi_fast_flow_length(item["flow_steps"]) is None:
            issue(issues, f"diagnostic: PI0-FAST autoregressive trace is empty, gapped, or malformed for {request_id}")
            break
        if directory.parent.name == "spatialvla" and spatialvla_flow_length(item["flow_steps"]) is None:
            issue(issues, f"diagnostic: SpatialVLA autoregressive trace is empty, gapped, or malformed for {request_id}")
            break
        if directory.parent.name == "univla_baai" and univla_flow_length(item["flow_steps"]) is None:
            issue(issues, f"diagnostic: UniVLA autoregressive trace is empty, gapped, or malformed for {request_id}")
            break
        if directory.parent.name == "rdt_1b":
            required_steps = {
                f"diffusion.{step}.{kind}"
                for step in range(5) for kind in ("prediction", "sample")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"diagnostic: RDT five-step prediction/sample trace incomplete for {request_id}")
                break
        if directory.parent.name == "cogact_small":
            required_steps = {f"diffusion.{step}.sample" for step in range(10)}
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"diagnostic: CogACT ten-step DDIM trace incomplete for {request_id}")
                break
        if directory.parent.name == "vitra_vla_3b":
            required_steps = {f"ddim.step_{step:02d}.state" for step in range(10)}
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"diagnostic: VITRA ten-step DDIM trace incomplete for {request_id}")
                break
        if directory.parent.name == "openhelix":
            required_steps = {f"flow.step_{step:02d}.state" for step in range(25)}
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"diagnostic: OpenHelix 25-step DDPM trace incomplete for {request_id}")
                break
        if directory.parent.name == "pi05":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10) for kind in ("velocity", "state")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"diagnostic: PI0.5 ten-step flow trace incomplete for {request_id}")
                break
        if directory.parent.name == "dexvla":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10) for kind in ("prediction", "state")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"diagnostic: DexVLA ten-step ScaleDP trace incomplete for {request_id}")
                break
        if mode == "batch_invariant" and (not output["exact"] or any(
            not step.get("exact") for step in item["flow_steps"].values()
        )):
            issue(issues, f"diagnostic: complete invariant path diverged for {request_id}, {arrangement}")
            break
    expected_groups = {(request_id, mode) for request_id in sources for mode in POLICY_MODES}
    if set(groups) != expected_groups or any(keys != required[mode] for (_, mode), keys in groups.items()):
        issue(issues, "diagnostic: 200 observations need every mode/arrangement for each request-local noise")
    fidelity_ids = set()
    fidelity_count = 0
    for item in stream_jsonl(fidelity):
        fidelity_count += 1
        request_id = item.get("request_id")
        if request_id not in sources or request_id in fidelity_ids:
            issue(issues, f"diagnostic: duplicate or unknown singleton-fidelity request {request_id}")
            break
        fidelity_ids.add(request_id)
        if item.get("replay_manifest_sha256") != manifest_hash or item.get("numerical_freeze_sha256") is not None or item.get("comparison") != "batch_invariant_singleton_vs_native_singleton":
            issue(issues, f"diagnostic: malformed singleton-fidelity provenance for {request_id}")
            break
        if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
            issue(issues, f"diagnostic: singleton fidelity uses another execution configuration for {request_id}")
            break
        if not required_metric_fields.issubset(item.get("output") or {}):
            issue(issues, f"diagnostic: incomplete singleton fidelity for {request_id}")
            break
        if directory.parent.name == "pi_fast" and not {
            "output.actions", "output.official_decode_valid",
        }.issubset((item.get("output", {}).get("tensors") or {}).keys()):
            issue(issues, f"diagnostic: PI0-FAST singleton fidelity omits decode validity for {request_id}")
            break
    if fidelity_ids != set(sources):
        issue(issues, f"diagnostic: {fidelity_count}/{expected_requests} singleton-fidelity records")
    expected_total = expected_requests * sum(map(len, required.values()))
    if total != expected_total:
        issue(issues, f"diagnostic: {total}/{expected_total} raw arrangements")
    return {"complete": len(issues) == start, "arrangements": total, "fidelity": fidelity_count}


def heldout_audit(directory: Path, issues: list[str], capacity: dict | None = None) -> dict:
    start = len(issues)
    invariance = directory / "heldout" / "invariance.jsonl"
    fidelity = directory / "heldout" / "singleton_fidelity.jsonl"
    if not invariance.is_file():
        invariance = invariance.with_suffix(invariance.suffix + ".gz")
    if not fidelity.is_file():
        fidelity = fidelity.with_suffix(fidelity.suffix + ".gz")
    if not invariance.is_file() or not fidelity.is_file():
        issue(issues, "heldout: missing raw invariance or singleton fidelity archive")
        return {"complete": False}
    manifest = read_json(directory / "replay_manifest.json") or {}
    multiplicity = request_multiplicity(manifest, issues)
    expected_requests = expected_replay_counts(directory.parent.name)[2] * multiplicity
    manifest_hash = sha256(directory / "replay_manifest.json") if (directory / "replay_manifest.json").is_file() else None
    freeze_hash = sha256(directory / "numerical_freeze.json") if (directory / "numerical_freeze.json").is_file() else None
    expected_ids = {
        request_id for item in manifest.get("observations", [])
        if item.get("split") == "heldout"
        for request_id in item.get("request_ids", [])
    }
    request_metadata = {
        request_id: (item.get("task"), item.get("episode_id"), item.get("observation_index"), noise_id)
        for item in manifest.get("observations", [])
        if item.get("split") == "heldout"
        for noise_id, request_id in enumerate(item.get("request_ids", []))
    }
    request_source = {
        request_id: (item, noise_id)
        for item in manifest.get("observations", [])
        if item.get("split") == "heldout"
        for noise_id, request_id in enumerate(item.get("request_ids", []))
    }
    if len(expected_ids) != expected_requests:
        issue(issues, f"heldout: manifest does not identify {expected_requests} held-out requests")
    keys = set()
    groups = defaultdict(set)
    total = 0
    nonsingleton = Counter()
    violations = Counter()
    request_violations = defaultdict(set)
    required_configuration = EXPECTED_EXECUTION_CONFIG.get(directory.parent.name)
    for item in stream_jsonl(invariance):
        total += 1
        if item.get("replay_manifest_sha256") != manifest_hash or item.get("numerical_freeze_sha256") != freeze_hash:
            issue(issues, "heldout: record mixes or omits replay/numerical freeze revision")
            break
        if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
            issue(issues, "heldout: adapter execution configuration differs from the repaired diagnostic path")
            break
        try:
            UUID(item.get("experiment_id", ""))
            recorded_time = datetime.fromisoformat(item.get("timestamp_utc", ""))
        except (ValueError, TypeError, AttributeError):
            issue(issues, "heldout: malformed experiment ID or UTC timestamp")
            break
        if recorded_time.tzinfo is None or recorded_time.utcoffset().total_seconds() != 0:
            issue(issues, "heldout: experiment timestamp is not UTC")
            break
        if not item.get("gpu") or not item.get("torch") or not item.get("dtype") or item.get("execution_mode") != "eager" or not isinstance(item.get("tf32_matmul"), bool) or not isinstance(item.get("tf32_cudnn"), bool):
            issue(issues, "heldout: GPU/precision/execution metadata missing")
            break
        if (item.get("task"), item.get("episode_id"), item.get("observation_index"), item.get("noise_id")) != request_metadata.get(item.get("request_id")):
            issue(issues, f"heldout: request metadata mismatch for {item.get('request_id')}")
            break
        source_for_hash = request_source.get(item.get("request_id"))
        if source_for_hash:
            source_row, source_noise_id = source_for_hash
            expected_noise_hash = source_row.get("noise", [])[source_noise_id]["sha256"] if multiplicity == 3 else None
            if item.get("source_image_sha256") != observation_image_sha256(source_row) or item.get("request_noise_sha256") != expected_noise_hash:
                issue(issues, f"heldout: real-input/noise hashes differ from frozen request {item.get('request_id')}")
                break
        key = (
            item.get("request_id"), item.get("implementation"), item.get("transformation"),
            item.get("logical_batch_size", item.get("batch_size")),
            item.get("target_batch_position"), item.get("companion_type"),
        )
        if key in keys:
            issue(issues, f"heldout: duplicate arrangement {key}")
        keys.add(key)
        groups[(item.get("request_id"), item.get("implementation"))].add(key[2:])
        if item.get("request_id") not in expected_ids:
            issue(issues, f"heldout: unknown or diagnostic request {item.get('request_id')}")
        logical_size = item.get("logical_batch_size", item.get("batch_size"))
        physical_size = item.get("batch_size")
        ordering = item.get("request_ordering", [])
        target_index = item.get("target_batch_index")
        if logical_size not in (1, 2, 4, 8, 16, 32) or not isinstance(physical_size, int) or len(ordering) != logical_size:
            issue(issues, f"heldout: invalid logical request set for {key}")
            break
        if not isinstance(target_index, int) or not 0 <= target_index < physical_size or ordering[target_index] != item.get("request_id"):
            issue(issues, f"heldout: target identity/index mismatch for {key}")
            break
        companion_ids = item.get("companion_ids", [])
        if companion_ids != [identifier for index, identifier in enumerate(ordering) if index != target_index]:
            issue(issues, f"heldout: companion IDs disagree with request ordering for {key}")
            break
        source = request_source.get(item.get("request_id"))
        if source:
            source_row, noise_id = source
            observations = manifest["observations"]
            expected_companions = [
                observations[index]["request_ids"][noise_id]
                for index in source_row.get("companion_replay_indices", [])[: logical_size - 1]
            ]
            if len(expected_companions) != logical_size - 1:
                issue(issues, f"heldout: frozen companion schedule incomplete for {key}")
                break
            if item.get("companion_type") == "duplicate":
                expected_companions = [item["request_id"]] * (logical_size - 1)
            if item.get("transformation") == "singleton":
                expected_ordering = [item["request_id"]]
            elif item.get("transformation") == "batch_position":
                expected_ordering = list(expected_companions)
                expected_ordering.insert(target_index, item["request_id"])
            elif item.get("transformation") == "permutation":
                expected_ordering = list(reversed([item["request_id"], *expected_companions]))
            else:
                expected_ordering = [item["request_id"], *expected_companions]
            if ordering != expected_ordering:
                issue(issues, f"heldout: batching schedule changed the frozen request set for {key}")
                break
        if item.get("companion_type") == "diverse" and (
            len(set(companion_ids)) != len(companion_ids) or item.get("request_id") in companion_ids
        ):
            issue(issues, f"heldout: diverse companions are not distinct for {key}")
            break
        if item.get("companion_type") == "duplicate" and any(identifier != item.get("request_id") for identifier in companion_ids):
            issue(issues, f"heldout: duplicate companions differ from target for {key}")
            break
        if item.get("transformation") == "partition":
            if physical_size != max(1, logical_size // 2) or item.get("partition_id") == "single_batch":
                issue(issues, f"heldout: partition did not execute a real split for {key}")
                break
        elif physical_size != logical_size:
            issue(issues, f"heldout: physical/logical batch mismatch for {key}")
            break
        output = item.get("output", {})
        if not all(field in output for field in ("exact", "unequal_element_count", "max_absolute_error", "mean_absolute_error", "rmse", "relative_l2_error", "nonfinite_count")):
            issue(issues, "heldout: a record lacks required numerical metrics")
            break
        if directory.parent.name == "pi_fast" and not {
            "output.actions", "output.official_decode_valid",
        }.issubset((output.get("tensors") or {}).keys()):
            issue(issues, f"heldout: PI0-FAST omits official decode validity for {key}")
            break
        if item.get("transformation") == "singleton" and not output["exact"]:
            issue(issues, f"heldout: singleton self-reference differs for {key}")
        if item.get("implementation") == "batch_invariant" and not output["exact"]:
            issue(issues, f"heldout: complete invariant path diverged for {key}")
        if physical_size > 1:
            mode = item.get("implementation")
            nonsingleton[mode] += 1
            if not output["exact"]:
                violations[mode] += 1
                request_violations[mode].add(item.get("request_id"))
        if "flow_steps" not in item:
            issue(issues, f"heldout: no iterative-step trace or explicit noniterative contract for {key}")
            break
        if directory.parent.name == "pi_fast":
            if pi_fast_flow_length(item["flow_steps"]) is None:
                issue(issues, f"heldout: PI0-FAST autoregressive trace is empty, gapped, or malformed for {key}")
                break
            if item.get("implementation") == "batch_invariant" and any(
                not step.get("exact") for step in item["flow_steps"].values()
            ):
                issue(issues, f"heldout: PI0-FAST invariant autoregressive step diverged for {key}")
                break
        if directory.parent.name == "spatialvla":
            if spatialvla_flow_length(item["flow_steps"]) is None:
                issue(issues, f"heldout: SpatialVLA autoregressive trace is empty, gapped, or malformed for {key}")
                break
            generation_steps = {
                label: step for label, step in item["flow_steps"].items()
                if label.startswith("generation.step_") and label.endswith(".scores")
            }
            if item.get("implementation") == "batch_invariant" and any(
                not step.get("exact") for step in generation_steps.values()
            ):
                issue(issues, f"heldout: SpatialVLA invariant autoregressive step diverged for {key}")
                break
        if directory.parent.name == "univla_baai":
            if univla_flow_length(item["flow_steps"]) is None:
                issue(issues, f"heldout: UniVLA autoregressive trace is empty, gapped, or malformed for {key}")
                break
            generation_steps = {
                label: step for label, step in item["flow_steps"].items()
                if label.startswith("flow.step_") and label.endswith(".scores")
            }
            if item.get("implementation") == "batch_invariant" and any(
                not step.get("exact") for step in generation_steps.values()
            ):
                issue(issues, f"heldout: UniVLA invariant autoregressive step diverged for {key}")
                break
        if directory.parent.name == "smolvla":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10)
                for kind in ("velocity", "state")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: SmolVLA ten-step flow trace incomplete for {key}")
                break
        if directory.parent.name == "pi05":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10)
                for kind in ("velocity", "state")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: PI0.5 ten-step flow trace incomplete for {key}")
                break
        if directory.parent.name == "dexvla":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10) for kind in ("prediction", "state")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: DexVLA ten-step ScaleDP trace incomplete for {key}")
                break
        if directory.parent.name == "xvla":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10)
                for kind in ("noisy_action", "prediction")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: X-VLA ten-step prediction trace incomplete for {key}")
                break
        if directory.parent.name in ("internvla_a15_standard", "internvla_a15_optimized"):
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10)
                for kind in ("state", "velocity")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: InternVLA ten-step action trace incomplete for {key}")
                break
        if directory.parent.name == "groot_n17":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(4)
                for kind in ("state", "velocity", "updated")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: GR00T four-step flow trace incomplete for {key}")
                break
        if directory.parent.name == "wsa_base":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10)
                for kind in ("state", "velocity")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: WSA ten-step flow trace incomplete for {key}")
                break
        if directory.parent.name == "uva":
            required_steps = {
                f"action.diffusion.step_{step:03d}" for step in range(100)
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: UVA 100-step action diffusion trace incomplete for {key}")
                break
        if directory.parent.name == "uva_joint":
            required_steps = {
                f"{series}.diffusion.step_{step:03d}"
                for series in ("action", "video") for step in range(100)
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: UVA joint 100-step action/video trace incomplete for {key}")
                break
        if directory.parent.name == "rdt_1b":
            required_steps = {
                f"diffusion.{step}.{kind}"
                for step in range(5) for kind in ("prediction", "sample")
            }
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: RDT five-step prediction/sample trace incomplete for {key}")
                break
        if directory.parent.name == "cogact_small":
            required_steps = {f"diffusion.{step}.sample" for step in range(10)}
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: CogACT ten-step DDIM trace incomplete for {key}")
                break
        if directory.parent.name == "vitra_vla_3b":
            required_steps = {f"ddim.step_{step:02d}.state" for step in range(10)}
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: VITRA ten-step DDIM trace incomplete for {key}")
                break
        if directory.parent.name == "openhelix":
            required_steps = {f"flow.step_{step:02d}.state" for step in range(25)}
            if not required_steps.issubset(item["flow_steps"]):
                issue(issues, f"heldout: OpenHelix 25-step DDPM trace incomplete for {key}")
                break
    capacity = capacity or {}
    required_by_mode = {
        mode: required_arrangements(replay_batch_sizes(directory.parent.name, capacity, mode))
        for mode in POLICY_MODES
    }
    expected = expected_requests * sum(len(required_by_mode[mode]) for mode in POLICY_MODES)
    if total != expected:
        issue(issues, f"heldout: {total}/{expected} required arrangement records")
    expected_groups = {(request_id, mode) for request_id in expected_ids for mode in POLICY_MODES}
    if set(groups) != expected_groups:
        issue(issues, "heldout: request/implementation group set differs from frozen manifest")
    for key in expected_groups:
        if groups.get(key, set()) != required_by_mode[key[1]]:
            issue(issues, f"heldout: {key} differs from the feasible frozen transformation matrix")
            if len(issues) > start + 20:
                break
    fidelity_rows = list(stream_jsonl(fidelity))
    if any(item.get("replay_manifest_sha256") != manifest_hash or item.get("numerical_freeze_sha256") != freeze_hash for item in fidelity_rows):
        issue(issues, "heldout: singleton fidelity mixes or omits frozen revision")
    for item in fidelity_rows:
        if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
            issue(issues, f"heldout: singleton fidelity uses another adapter execution configuration for {item.get('request_id')}")
            break
        if item.get("comparison") != "batch_invariant_singleton_vs_native_singleton" or not all(field in item.get("output", {}) for field in ("exact", "max_absolute_error", "rmse", "relative_l2_error")):
            issue(issues, f"heldout: singleton fidelity metrics/label missing for {item.get('request_id')}")
            break
        if directory.parent.name == "pi_fast" and not {
            "output.actions", "output.official_decode_valid",
        }.issubset((item.get("output", {}).get("tensors") or {}).keys()):
            issue(issues, f"heldout: PI0-FAST singleton fidelity omits decode validity for {item.get('request_id')}")
            break
        if (item.get("task"), item.get("episode_id"), item.get("observation_index"), item.get("noise_id")) != request_metadata.get(item.get("request_id")):
            issue(issues, f"heldout: singleton fidelity request metadata mismatch for {item.get('request_id')}")
            break
        source = request_source.get(item.get("request_id"))
        if source:
            source_row, source_noise_id = source
            expected_noise_hash = source_row.get("noise", [])[source_noise_id]["sha256"] if multiplicity == 3 else None
            if item.get("source_image_sha256") != observation_image_sha256(source_row) or item.get("request_noise_sha256") != expected_noise_hash:
                issue(issues, f"heldout: singleton fidelity source/noise hash mismatch for {item.get('request_id')}")
                break
    fidelity_ids = [item.get("request_id") for item in fidelity_rows]
    if len(fidelity_ids) != expected_requests or set(fidelity_ids) != expected_ids:
        issue(issues, f"heldout: {len(fidelity_ids)}/{expected_requests} unique singleton fidelity records")
    return {
        "complete": len(issues) == start,
        "arrangements": total,
        "fidelity": len(fidelity_ids),
        "nonsingleton": dict(nonsingleton),
        "violations": dict(violations),
        "request_violations": {mode: len(ids) for mode, ids in request_violations.items()},
    }


def operator_audit(path: Path, issues: list[str], capacity: dict | None = None) -> dict:
    start = len(issues)
    if not path.is_file():
        issue(issues, f"operators: missing {path}")
        return {"complete": False}
    manifest = read_json(path.parent / "configurations.json")
    if not manifest or not manifest.get("configurations"):
        issue(issues, "operators: missing real-shape configurations.json")
        return {"complete": False}
    model_id = path.parents[2].name
    if (model_id in EXPECTED_EXECUTION_CONFIG
            and manifest.get("adapter_execution_configuration") != EXPECTED_EXECUTION_CONFIG[model_id]):
        issue(issues, "operators: adapter execution configuration differs from the audited policy path")
    replay_path = path.parents[1] / "replay_manifest.json"
    replay_hash = sha256(replay_path) if replay_path.is_file() else None
    for batch_size in (1, 8):
        trace_path = path.parent / f"aten_trace_B{batch_size}.json"
        trace = read_json(trace_path)
        if (not trace or trace.get("batch_size") != batch_size
                or len(trace.get("request_ids") or []) != batch_size
                or trace.get("replay_manifest_sha256") != replay_hash):
            issue(issues, f"operators: B{batch_size} ATen trace is missing or uses another frozen replay")
            continue
        if (model_id in EXPECTED_EXECUTION_CONFIG
                and trace.get("adapter_execution_configuration") != EXPECTED_EXECUTION_CONFIG[model_id]):
            issue(issues, f"operators: B{batch_size} ATen trace uses another execution configuration")
        if model_id == "dexvla" and trace.get("implementation") != (
            "native" if batch_size == 1 else "batch_invariant"
        ):
            issue(issues, f"operators: DexVLA B{batch_size} trace uses the wrong native/repaired path")
    manifest_hash = sha256(path.parent / "configurations.json")
    by_id = {config["id"]: config for config in manifest["configurations"]}
    if len(by_id) != len(manifest["configurations"]):
        issue(issues, "operators: duplicate real-shape configuration IDs")
    checkpoint_weights = manifest.get("checkpoint_weights")
    if checkpoint_weights:
        weights_path = Path(checkpoint_weights)
        if not weights_path.is_file() or sha256(weights_path) != manifest.get("checkpoint_weights_sha256"):
            issue(issues, "operators: source checkpoint weights missing or hash-changed")
        elif weights_path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(weights_path, framework="pt", device="cpu") as weights:
                for config in manifest["configurations"]:
                    key = config.get("source_weight_key")
                    if key and (key not in weights.keys() or weights.get_slice(key).get_shape() != config.get("weight_shape")):
                        issue(issues, f"operators: source checkpoint weight shape differs for {config.get('id')}")
        elif any(config.get("source_weight_key") for config in manifest["configurations"]):
            issue(issues, "operators: non-safetensors checkpoint cannot substantiate source weight keys")
    elif model_id == "openhelix":
        receipt_path = Path(manifest.get("checkpoint_stream_receipt", ""))
        receipt = read_json(receipt_path) if receipt_path.is_file() else None
        expected_files = {
            Path(name).name: {"bytes": size, "sha256": digest}
            for name, size, digest in OPENHELIX_CHECKPOINT_FILES
        }
        actual_files = {
            name: {"bytes": record.get("bytes"), "sha256": record.get("sha256")}
            for name, record in (receipt or {}).get("files", {}).items()
        }
        planner = (receipt or {}).get("strict_planner_load") or {}
        policy = (receipt or {}).get("strict_policy_load") or {}
        if (
            not receipt
            or sha256(receipt_path) != manifest.get("checkpoint_stream_receipt_sha256")
            or receipt.get("repository") != "OpenHelix/openhelix"
            or receipt.get("revision") != OPENHELIX_CHECKPOINT_REVISION
            or receipt.get("all_pinned_files_verified") is not True
            or actual_files != expected_files
            or planner.get("missing_named_tensors") != []
            or planner.get("meta_parameters_remaining") is not False
            or policy.get("missing_keys") != []
            or policy.get("unexpected_keys") != []
        ):
            issue(issues, "operators: OpenHelix exact streamed checkpoint receipt is incomplete")
    elif manifest.get("checkpoint_weight_shards"):
        shards = manifest["checkpoint_weight_shards"]
        hashes = manifest.get("checkpoint_weight_shards_sha256")
        if (not isinstance(shards, list) or not shards
                or not isinstance(hashes, list) or len(hashes) != len(shards)):
            issue(issues, "operators: malformed checkpoint shard provenance")
        else:
            shard_paths = [Path(value) for value in shards]
            if any(not shard.is_file() for shard in shard_paths) or any(
                sha256(shard) != expected for shard, expected in zip(shard_paths, hashes)
            ):
                issue(issues, "operators: source checkpoint shards missing or hash-changed")
            else:
                from safetensors import safe_open

                required_weights = {
                    config["source_weight_key"]: (config.get("id"), config.get("weight_shape"))
                    for config in manifest["configurations"] if config.get("source_weight_key")
                }
                found = defaultdict(list)
                for shard in shard_paths:
                    with safe_open(shard, framework="pt", device="cpu") as weights:
                        for key in required_weights.keys() & set(weights.keys()):
                            found[key].append(weights.get_slice(key).get_shape())
                for key, (configuration_id, expected_shape) in required_weights.items():
                    if found.get(key) != [expected_shape]:
                        issue(issues, f"operators: source checkpoint shard weight shape differs for {configuration_id}")
    expected = set()
    # This is the operator-level sweep from the π0 plan, independent of policy
    # batch capacity. A policy OOM is not evidence that a standalone operator
    # configuration cannot run at that batch size.
    required_sizes = (1, 2, 4, 8, 16, 31, 32, 33, 64)
    for config in manifest["configurations"]:
        operation = config.get("operation")
        sdpa_query_heads = config.get("query_heads", config.get("heads"))
        sdpa_key_heads = config.get("key_heads", config.get("heads"))
        sdpa_value_heads = config.get("value_heads", config.get("heads"))
        sdpa_q_stride = config.get("source_q_stride", config.get("source_qkv_stride"))
        sdpa_k_stride = config.get("source_k_stride", config.get("source_qkv_stride"))
        sdpa_v_stride = config.get("source_v_stride", config.get("source_qkv_stride"))
        valid_shape = (
            all(config.get(field) for field in ("m_per_request", "k", "n"))
            if operation in ("mm", "addmm") else
            all(config.get(field) for field in ("m_per_request", "m", "k", "n"))
            if operation == "bmm" else
            (all(config.get(field) for field in ("m_per_request", "weight_shape", "stride", "padding"))
             and bool(config.get("input_per_unit", config.get("input_per_image"))))
            if operation in ("conv1d", "conv2d", "conv3d", "conv_transpose2d", "conv_transpose3d") else
            (bool(config.get("m_per_request")) and bool(config.get("input_per_unit"))
             and isinstance(config.get("dim"), int)
             and (operation == "softmax" or isinstance(config.get("keepdim"), bool))
             and config.get("layout") == "contiguous")
            if operation in ("mean", "softmax") else
            all(config.get(field) for field in (
                "m_per_request", "query_length", "key_length", "head_dim",
                "value_dim", "scale", "source_trace_path",
            )) and all((sdpa_query_heads, sdpa_key_heads, sdpa_value_heads,
                       sdpa_q_stride, sdpa_k_stride, sdpa_v_stride))
            and sdpa_key_heads == sdpa_value_heads
            and sdpa_query_heads % sdpa_key_heads == 0
            and bool(config.get("enable_gqa", False)) == (sdpa_query_heads != sdpa_key_heads)
            if operation == "sdpa" else False
        )
        if not config.get("source_operation") or not config.get("layout") or not valid_shape:
            issue(issues, f"operators: incomplete source-shape manifest {config.get('id')}")
        if operation == "sdpa" and valid_shape:
            trace_path = Path(config["source_trace_path"])
            trace = read_json(trace_path) if trace_path.is_file() else None
            wanted_shapes = (
                [8 * config["m_per_request"], sdpa_query_heads, config["query_length"], config["head_dim"]],
                [8 * config["m_per_request"], sdpa_key_heads, config["key_length"], config["head_dim"]],
                [8 * config["m_per_request"], sdpa_value_heads, config["key_length"], config["value_dim"]],
            )
            wanted_strides = (sdpa_q_stride, sdpa_k_stride, sdpa_v_stride)
            if not trace or not any(
                row.get("aten_operation") == "aten::scaled_dot_product_attention"
                and tuple(argument.get("shape") for argument in row.get("arguments", [])[:3]) == wanted_shapes
                and tuple(argument.get("stride") for argument in row.get("arguments", [])[:3]) == wanted_strides
                for row in trace.get("operations", [])
            ):
                issue(issues, f"operators: {config['id']} SDPA shape/stride is not in the pinned ATen trace")
        batch_sizes = config.get("batch_sizes", required_sizes)
        if not set(required_sizes).issubset(batch_sizes):
            issue(issues, f"operators: {config.get('id')} lacks required feasible sizes {required_sizes}")
        for batch_size in batch_sizes:
            default_positions = ("first",) if batch_size == 1 else ("first", "last")
            positions = config.get("target_positions_by_batch", {}).get(str(batch_size), default_positions)
            for position in positions:
                expected.add(f"{config['id']}|B{batch_size}|{position}")
    trace_path = path.parent / "aten_trace_B8.json"
    if trace_path.is_file():
        trace = read_json(trace_path) or {}
        observed_sdpa = {
            tuple((tuple(arg.get("shape", [])), tuple(arg.get("stride", []))) for arg in row.get("arguments", [])[:3])
            for row in trace.get("operations", [])
            if row.get("aten_operation") == "aten::scaled_dot_product_attention"
            and len(row.get("arguments", [])) >= 3
        }
        configured_sdpa = {
            (
                ((8 * config["m_per_request"], config.get("query_heads", config.get("heads")), config["query_length"], config["head_dim"]), tuple(config.get("source_q_stride", config.get("source_qkv_stride")))),
                ((8 * config["m_per_request"], config.get("key_heads", config.get("heads")), config["key_length"], config["head_dim"]), tuple(config.get("source_k_stride", config.get("source_qkv_stride")))),
                ((8 * config["m_per_request"], config.get("value_heads", config.get("heads")), config["key_length"], config["value_dim"]), tuple(config.get("source_v_stride", config.get("source_qkv_stride")))),
            )
            for config in manifest["configurations"] if config.get("operation") == "sdpa"
        }
        if observed_sdpa != configured_sdpa:
            issue(issues, f"operators: {len(configured_sdpa)}/{len(observed_sdpa)} traced SDPA shapes/strides are qualified")
    for evidence in manifest.get("source_evidence", []):
        if not Path(evidence).is_file():
            issue(issues, f"operators: source-shape evidence missing: {evidence}")
    seeds = defaultdict(set)
    for item in stream_jsonl(path):
        key = item.get("operator_configuration")
        seed = item.get("seed")
        if not isinstance(seed, int) or item.get("generator_seed") != 1_000_000 + seed:
            issue(issues, f"operators: fixed generator seed mismatch for {key}: {seed}")
        required_fields = (
            "source_operation", "dispatched_aten_operation", "left_shape", "right_shape",
            "left_stride", "right_stride", "dtype", "layout", "native_batch_invariance",
            "invariant_batch_invariance", "reference_batch_invariance",
            "singleton_fidelity", "reference_singleton_fidelity",
            "native_latency_seconds", "invariant_latency_seconds", "reference_latency_seconds",
        )
        if any(field not in item for field in required_fields):
            issue(issues, f"operators: missing dispatch/layout/numerical/performance evidence for {key}, seed {seed}")
            break
        if seed in seeds[key]:
            issue(issues, f"operators: duplicate seed {seed} for {key}")
        seeds[key].add(seed)
        if key not in expected:
            issue(issues, f"operators: unplanned configuration {key}")
        config = by_id.get(str(key).split("|", 1)[0], {})
        if item.get("configuration_sha256") != manifest_hash or item.get("source_operation") != config.get("source_operation") or item.get("layout") != config.get("layout"):
            issue(issues, f"operators: source-shape provenance mismatch for {key}")
        if config.get("operation") in ("mm", "addmm"):
            batch_size = item.get("batch_size")
            if not isinstance(batch_size, int) or item.get("dispatched_aten_operation") != f"aten::{config['operation']}" or item.get("left_shape") != [batch_size * config["m_per_request"], config["k"]] or item.get("right_shape") != [config["k"], config["n"]]:
                issue(issues, f"operators: matrix shape/dispatch differs from real-shape manifest for {key}")
        elif config.get("operation") in ("mean", "softmax"):
            batch_size = item.get("batch_size")
            shape = [batch_size * config["m_per_request"], *config["input_per_unit"]] if isinstance(batch_size, int) else None
            expected_stride = [1] * len(shape) if shape else None
            if shape:
                for index in range(len(shape) - 2, -1, -1):
                    expected_stride[index] = expected_stride[index + 1] * shape[index + 1]
            expected_dispatch = (
                "aten::mean" if config.get("operation") == "mean" else "aten::_softmax"
            )
            if (item.get("dispatched_aten_operation") != expected_dispatch
                    or item.get("left_shape") != shape or item.get("right_shape") is not None
                    or item.get("left_stride") != expected_stride or item.get("right_stride") is not None):
                issue(issues, f"operators: reduction shape/dispatch differs from configuration for {key}")
        elif config.get("operation") in ("conv1d", "conv2d", "conv3d", "conv_transpose2d", "conv_transpose3d"):
            batch_size = item.get("batch_size")
            input_shape = config.get("input_per_unit", config.get("input_per_image"))
            shape = [batch_size * config["m_per_request"], *input_shape] if isinstance(batch_size, int) else None
            expected_dispatch = (
                f"aten::{config['operation']}"
                if config.get("operation") in ("conv_transpose2d", "conv_transpose3d") else
                "aten::convolution"
            )
            if item.get("dispatched_aten_operation") != expected_dispatch or item.get("left_shape") != shape or item.get("right_shape") != config["weight_shape"]:
                issue(issues, f"operators: convolution shape/dispatch differs from configuration for {key}")
            if shape:
                expected_stride = [1] * len(shape)
                for index in range(len(shape) - 2, -1, -1):
                    expected_stride[index] = expected_stride[index + 1] * shape[index + 1]
                if config.get("layout") == "channels_last":
                    _, channels, height, width = shape
                    expected_stride = [height * width * channels, 1, width * channels, channels]
                elif config.get("layout") == "channels_last_3d":
                    _, channels, depth, height, width = shape
                    expected_stride = [
                        depth * height * width * channels, 1,
                        height * width * channels, width * channels, channels,
                    ]
                if item.get("left_stride") != expected_stride:
                    issue(issues, f"operators: convolution input layout differs from configuration for {key}")
        elif config.get("operation") == "bmm":
            batch_size = item.get("batch_size")
            heads = batch_size * config["m_per_request"] if isinstance(batch_size, int) else None
            if item.get("dispatched_aten_operation") != "aten::bmm" or item.get("left_shape") != [heads, config["m"], config["k"]] or item.get("right_shape") != [heads, config["k"], config["n"]]:
                issue(issues, f"operators: BMM shape/dispatch differs from real-shape manifest for {key}")
            m, k, n = config["m"], config["k"], config["n"]
            expected_left_stride = [m * k * 2, k * 2, 2] if config.get("layout") == "left_strided" else [m * k, k, 1]
            expected_right_stride = [n * k, 1, k] if config.get("layout") == "right_transposed" else [k * n, n, 1]
            if item.get("left_stride") != expected_left_stride or item.get("right_stride") != expected_right_stride:
                issue(issues, f"operators: BMM operand layout differs from configuration for {key}")
        elif config.get("operation") == "sdpa":
            batch_size = item.get("batch_size")
            if isinstance(batch_size, int):
                outer = batch_size * config["m_per_request"]
                query_shape = [outer, config.get("query_heads", config.get("heads")), config["query_length"], config["head_dim"]]
                key_shape = [outer, config.get("key_heads", config.get("heads")), config["key_length"], config["head_dim"]]
                value_shape = [outer, config.get("value_heads", config.get("heads")), config["key_length"], config["value_dim"]]
            else:
                query_shape = key_shape = value_shape = None
            if (item.get("dispatched_aten_operation") != "aten::scaled_dot_product_attention"
                    or item.get("left_shape") != query_shape or item.get("right_shape") != key_shape
                    or item.get("value_shape") != value_shape
                    or item.get("left_stride") != config.get("source_q_stride", config.get("source_qkv_stride"))
                    or item.get("right_stride") != config.get("source_k_stride", config.get("source_qkv_stride"))
                    or item.get("value_stride") != config.get("source_v_stride", config.get("source_qkv_stride"))):
                issue(issues, f"operators: SDPA actual Q/K/V shape or stride differs from trace for {key}")
        if not item.get("invariant_batch_invariance", {}).get("exact"):
            issue(issues, f"operators: invariant implementation not exact at {key}, seed {seed}")
        if not item.get("reference_batch_invariance", {}).get("exact"):
            issue(issues, f"operators: explicit per-matrix reference not exact at {key}, seed {seed}")
        if not all(item.get(field, 0) > 0 for field in (
            "native_latency_seconds", "invariant_latency_seconds", "reference_latency_seconds"
        )):
            issue(issues, f"operators: missing native/reference/invariant positive latency at {key}, seed {seed}")
        if seed == 0:
            profile = item.get("dispatch_profile") or {}
            if item.get("dispatched_aten_operation") not in profile.get("aten_events", []) or not profile.get("cuda_kernels"):
                issue(issues, f"operators: dispatched ATen/CUDA profile missing at {key}")
        if seed in (0, 1, 2) and not item.get("fp64_reference"):
            issue(issues, f"operators: missing selected FP64 reference at {key}, seed {seed}")
    if set(seeds) != expected or any(values != set(range(OPERATOR_SEEDS)) for values in seeds.values()):
        issue(issues, "operators: each real-shape configuration needs 100 unique seeds")
    return {"complete": len(issues) == start, "configurations": len(seeds)}


def queue_audit(
    path: Path, issues: list[str], *, logical_batch_size: int = 4,
    fixed_batch_size: int = 8, serving_timings_path: Path | None = None,
) -> dict:
    start = len(issues)
    if not path.is_file():
        issue(issues, f"serving queue: missing {path}")
        return {"complete": False}
    audited_capacity = None
    expected_selection = None
    timed_fixed_shape = None
    if serving_timings_path is not None:
        if not serving_timings_path.is_file():
            issue(issues, f"serving queue: reference timing file missing {serving_timings_path}")
            return {"complete": False}
        timing_rows = list(stream_jsonl(serving_timings_path))
        environments = [row for row in timing_rows if row.get("record_type") == "environment"]
        fixed_shapes = [row for row in timing_rows if row.get("record_type") == "fixed_shape"]
        if len(environments) != 1:
            issue(issues, "serving queue: expected one serving-timing environment record")
            return {"complete": False}
        if len(fixed_shapes) != 1:
            issue(issues, "serving queue: expected one fixed-padded timing shape record")
            return {"complete": False}
        timed_fixed_shape = fixed_shapes[0].get("metadata")
        expected_selection = environments[0].get("input_selection")
        if expected_selection is None:
            issue(issues, "serving queue: serving-timing input selection is missing")
            return {"complete": False}
        timing_input_selection_audit(expected_selection, fixed_batch_size, issues)
        if len(issues) > start:
            return {"complete": False}
        native_latencies = [
            row["latency_seconds"] for row in timing_rows
            if row.get("record_type") == "timing" and row.get("configuration") == "native_dynamic"
        ]
        if len(native_latencies) != SESSIONS * TIMED_CALLS:
            issue(issues, "serving queue: reference capacity lacks 1,000 native-dynamic timing calls")
            return {"complete": False}
        audited_capacity = logical_batch_size / statistics.median(native_latencies)
    groups = Counter()
    keys = set()
    batches = defaultdict(list)
    rates = defaultdict(set)
    latencies = defaultdict(list)
    arrivals = defaultdict(list)
    completions = defaultdict(list)
    peak_bytes = defaultdict(list)
    numerical_violations = Counter()
    numerical_nonexact = Counter()
    fields = (
        "request_id", "arrival_time", "batch_assignment", "batch_occupancy",
        "service_start", "completion_time", "queueing_latency_seconds",
        "service_latency_seconds", "arrival_to_completion_seconds", "numerical_contract",
        "peak_incremental_bytes",
    )
    required_configuration = EXPECTED_EXECUTION_CONFIG.get(path.parents[2].name)
    for item in stream_jsonl(path):
        if item.get("configuration") == "fixed_size_padded" and item.get("fixed_shape_metadata") != timed_fixed_shape:
            issue(issues, "serving queue: fixed-padded input shape differs from timing evidence")
            break
        if expected_selection is not None:
            fixture_index = item.get("fixture_index")
            if (item.get("input_selection") != expected_selection
                    or not isinstance(fixture_index, int) or not 0 <= fixture_index < fixed_batch_size
                    or item.get("source_replay_index") != expected_selection["replay_indices"][fixture_index]
                    or item.get("source_request_id") != expected_selection["request_ids"][fixture_index]):
                issue(issues, "serving queue: source request differs from pinned timing input selection")
                break
        if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
            issue(issues, "serving queue: adapter execution configuration differs from repaired policy path")
            break
        group = (item.get("configuration"), item.get("offered_load"))
        key = (*group, item.get("request_id"))
        if key in keys:
            issue(issues, f"serving queue: duplicate request {key}")
        keys.add(key)
        groups[group] += 1
        batches[(*group, item.get("batch_assignment"))].append(item)
        latencies[group].append(item.get("arrival_to_completion_seconds"))
        arrivals[group].append(item.get("arrival_time"))
        completions[group].append(item.get("completion_time"))
        numerical_violations[group] += item.get("numerical_contract") == "violated"
        numerical_nonexact[group] += item.get("numerical_contract") != "exact"
        rate = item.get("offered_rate_requests_per_second")
        capacity = item.get("reference_capacity_requests_per_second")
        if not isinstance(rate, (int, float)) or not isinstance(capacity, (int, float)) or rate <= 0 or capacity <= 0:
            issue(issues, f"serving queue: missing positive offered-rate/capacity evidence for {key}")
            break
        if audited_capacity is not None and abs(capacity - audited_capacity) > max(1e-5, audited_capacity * 1e-4):
            issue(issues, f"serving queue: capacity differs from audited native service timing for {key}")
            break
        rates[group].add(rate / capacity)
        if any(field not in item for field in fields):
            issue(issues, f"serving queue: missing request field in {key}")
            break
        arrival = item["arrival_time"]
        start_time = item["service_start"]
        completion = item["completion_time"]
        if not all(isinstance(value, (int, float)) for value in (arrival, start_time, completion)) or not arrival <= start_time <= completion:
            issue(issues, f"serving queue: invalid event ordering for {key}")
            break
        if item["batch_occupancy"] < 1 or item["queueing_latency_seconds"] < 0 or item["service_latency_seconds"] <= 0:
            issue(issues, f"serving queue: invalid occupancy or latency for {key}")
            break
        occupancy = item["batch_occupancy"]
        expected_physical = (
            fixed_batch_size if item.get("configuration") == "fixed_size_padded"
            else 1 if item.get("configuration") == "serial_singleton"
            else occupancy
        )
        target_position = item.get("target_batch_position")
        if occupancy > logical_batch_size or item.get("physical_batch_size") != expected_physical or not isinstance(target_position, int) or not 0 <= target_position < occupancy:
            issue(issues, f"serving queue: invalid logical/physical batch shape for {key}")
            break
        if not isinstance(item["peak_incremental_bytes"], int) or item["peak_incremental_bytes"] < 0:
            issue(issues, f"serving queue: invalid peak memory for {key}")
            break
        peak_bytes[group].append(item["peak_incremental_bytes"])
        if abs((completion - arrival) - item["arrival_to_completion_seconds"]) > 1e-5:
            issue(issues, f"serving queue: end-to-end latency arithmetic mismatch for {key}")
            break
        if abs((start_time - arrival) - item["queueing_latency_seconds"]) > 1e-5 or abs((completion - start_time) - item["service_latency_seconds"]) > 1e-5:
            issue(issues, f"serving queue: queue/service latency arithmetic mismatch for {key}")
            break
        if item["numerical_contract"] not in ("exact", "violated", "reference_only"):
            issue(issues, f"serving queue: unverified numerical contract for {key}")
            break
    if len(issues) > start:
        return {"complete": False, "requests": sum(groups.values()), "groups": len(groups)}
    expected = {(mode, load) for mode in SERVING_MODES for load in LOADS}
    if set(groups) != expected or any(groups[group] < 250 for group in expected):
        issue(issues, "serving queue: require four loads x four modes x at least 250 requests")
    for batch_key, items in batches.items():
        if len(items) != items[0].get("batch_occupancy") or len({item.get("service_start") for item in items}) != 1 or len({item.get("completion_time") for item in items}) != 1:
            issue(issues, f"serving queue: inconsistent shared batch assignment {batch_key}")
            break
    bands = {
        "underload": (0.0, 0.5), "moderate": (0.4, 0.8),
        "near_saturation": (0.8, 1.1), "saturation": (1.1, float("inf")),
    }
    for group in expected:
        values = rates.get(group, set())
        low, high = bands[group[1]]
        if len(values) != 1 or not low < next(iter(values), 0.0) < high:
            issue(issues, f"serving queue: offered load {group} is outside its required band")
        if group[0] in ("batch_invariant_dynamic", "serial_singleton") and numerical_nonexact[group]:
            issue(issues, f"serving queue: {group} lacks an exact singleton numerical contract")
    return {
        "complete": len(issues) == start,
        "requests": sum(groups.values()),
        "groups": len(groups),
        "group_counts": {f"{mode}|{load}": count for (mode, load), count in sorted(groups.items())},
        "group_metrics": {
            f"{mode}|{load}": {
                "throughput_requests_per_second": count / (max(completions[mode, load]) - min(arrivals[mode, load])),
                "median_arrival_to_completion_seconds": statistics.median(latencies[mode, load]),
                "p95_arrival_to_completion_seconds": (
                    lambda ordered, position: ordered[int(position)]
                    + (ordered[min(int(position) + 1, len(ordered) - 1)] - ordered[int(position)])
                    * (position - int(position))
                )(sorted(latencies[mode, load]), 0.95 * (count - 1)),
                "occupancy_distribution": dict(Counter(
                    len(items) for (batch_mode, batch_load, _), items in batches.items()
                    if (batch_mode, batch_load) == (mode, load)
                )),
                "mean_peak_incremental_bytes": statistics.fmean(peak_bytes[mode, load]),
                "numerical_violations": numerical_violations[mode, load],
            }
            for (mode, load), count in sorted(groups.items())
        },
    }


def queue_summary_audit(path: Path, raw_counts: dict, raw_metrics: dict, issues: list[str]) -> dict:
    start = len(issues)
    report = read_json(path)
    if not report:
        issue(issues, f"serving queue: missing summary {path}")
        return {"complete": False}
    rows = report.get("groups", [])
    counts = {f"{row.get('configuration')}|{row.get('offered_load')}": row.get("requests") for row in rows}
    if counts != raw_counts or len(rows) != 16:
        issue(issues, "serving queue: summary group counts disagree with raw requests")
    for row in rows:
        if any(field not in row for field in (
            "throughput_requests_per_second", "median_arrival_to_completion_seconds",
            "p95_arrival_to_completion_seconds", "occupancy_distribution",
            "mean_peak_incremental_bytes", "numerical_violations",
        )):
            issue(issues, "serving queue: summary lacks throughput/latency/occupancy/memory/contract")
            break
        key = f"{row.get('configuration')}|{row.get('offered_load')}"
        raw = raw_metrics.get(key)
        if raw and (
            abs(row["throughput_requests_per_second"] - raw["throughput_requests_per_second"]) > 1e-5
            or abs(row["median_arrival_to_completion_seconds"] - raw["median_arrival_to_completion_seconds"]) > 1e-5
            or row["numerical_violations"] != raw["numerical_violations"]
            or abs(row["p95_arrival_to_completion_seconds"] - raw["p95_arrival_to_completion_seconds"]) > 1e-5
            or {int(occupancy): count for occupancy, count in row["occupancy_distribution"].items()} != raw["occupancy_distribution"]
            or abs(row["mean_peak_incremental_bytes"] - raw["mean_peak_incremental_bytes"]) > 1e-5
        ):
            issue(issues, f"serving queue: summary disagrees with raw service/request records for {key}")
    return {"complete": len(issues) == start, "groups": len(rows)}


def ablation_audit(path: Path, issues: list[str]) -> dict:
    start = len(issues)
    if not path.is_file():
        issue(issues, f"ablation: missing {path}")
        return {"complete": False}
    manifest = read_json(path.parent.parent / "replay_manifest.json") or {}
    manifest_path = path.parent.parent / "replay_manifest.json"
    manifest_hash = sha256(manifest_path) if manifest_path.is_file() else None
    model_id = path.parents[2].name
    required_configuration = EXPECTED_EXECUTION_CONFIG.get(model_id)
    multiplicity = request_multiplicity(manifest, issues)
    expected_ids = {
        request_id for item in manifest.get("observations", [])
        if item.get("split") == "diagnostic"
        for request_id in item.get("request_ids", [])
    }
    request_source = {
        request_id: (item, noise_id)
        for item in manifest.get("observations", [])
        if item.get("split") == "diagnostic"
        for noise_id, request_id in enumerate(item.get("request_ids", []))
    }
    if len(expected_ids) != 200 * multiplicity:
        issue(issues, "ablation: diagnostic request set differs from frozen replay manifest")
    modes = (
        "native", "native_deterministic", "existing_invariant_ops",
        "operators_plus_patch_projection", "explicit_per_matrix_reference", "full_invariant",
    )
    records = defaultdict(set)
    inapplicable = set()
    for item in stream_jsonl(path):
        mode = item.get("mode")
        if mode not in modes:
            issue(issues, f"ablation: unexpected mode {mode}")
            continue
        if item.get("status") == "inapplicable":
            evidence = item.get("evidence_path")
            if mode in ("native", "full_invariant") or not item.get("reason") or not evidence or not Path(evidence).is_file():
                issue(issues, f"ablation: unsupported inapplicability for {mode}")
            inapplicable.add(mode)
            continue
        request_id = item.get("request_id")
        if request_id not in expected_ids:
            issue(issues, f"ablation: unknown or held-out request for {mode}: {request_id}")
        if not request_id or request_id in records[mode]:
            issue(issues, f"ablation: missing/duplicate request for {mode}: {request_id}")
        records[mode].add(request_id)
        source = request_source.get(request_id)
        required_metrics = {
            "exact", "unequal_element_count", "max_absolute_error", "mean_absolute_error",
            "rmse", "relative_l2_error", "nonfinite_count",
        }
        output = item.get("output") or {}
        flow_steps = item.get("flow_steps") or {}
        if (item.get("replay_manifest_sha256") != manifest_hash or source is None
                or not required_metrics.issubset(output)
                or not isinstance(flow_steps, dict) or not flow_steps
                or any(not required_metrics.issubset(value) for value in flow_steps.values())):
            issue(issues, f"ablation: missing divergence/output/step metrics for {mode}: {request_id}")
            break
        if (required_configuration is not None
                and item.get("adapter_execution_configuration") != required_configuration):
            issue(issues, f"ablation: execution configuration differs for {mode}: {request_id}")
            break
        if path.parents[2].name == "pi_fast" and not {
            "output.actions", "output.official_decode_valid",
        }.issubset((output.get("tensors") or {}).keys()):
            issue(issues, f"ablation: PI0-FAST omits official decode validity for {mode}: {request_id}")
            break
        if path.parents[2].name == "pi_fast" and pi_fast_flow_length(flow_steps) is None:
            issue(issues, f"ablation: PI0-FAST autoregressive trace is empty, gapped, or malformed for {mode}: {request_id}")
            break
        if path.parents[2].name == "spatialvla" and spatialvla_flow_length(flow_steps) is None:
            issue(issues, f"ablation: SpatialVLA autoregressive trace is empty, gapped, or malformed for {mode}: {request_id}")
            break
        if path.parents[2].name == "univla_baai" and univla_flow_length(flow_steps) is None:
            issue(issues, f"ablation: UniVLA autoregressive trace is empty, gapped, or malformed for {mode}: {request_id}")
            break
        if path.parents[2].name == "pi05":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10) for kind in ("velocity", "state")
            }
            if not required_steps.issubset(flow_steps):
                issue(issues, f"ablation: PI0.5 ten-step flow trace incomplete for {mode}: {request_id}")
                break
        if path.parents[2].name == "dexvla":
            required_steps = {
                f"flow.step_{step:02d}.{kind}"
                for step in range(10) for kind in ("prediction", "state")
            }
            if not required_steps.issubset(flow_steps):
                issue(issues, f"ablation: DexVLA ten-step ScaleDP trace incomplete for {mode}: {request_id}")
                break
        row, noise_id = source
        expected_noise_hash = (
            row.get("noise", [])[noise_id]["sha256"]
            if len(row.get("noise", [])) == 3 else None
        )
        if ((item.get("task"), item.get("episode_id"), item.get("observation_index"),
             item.get("noise_id")) !=
                (row.get("task"), row.get("episode_id"), row.get("observation_index"), noise_id)
                or item.get("source_image_sha256") != observation_image_sha256(row)
                or item.get("request_noise_sha256") != expected_noise_hash):
            issue(issues, f"ablation: frozen request provenance differs for {mode}: {request_id}")
            break
        # JSONL records are emitted with sort_keys=True, so dictionary order is
        # lexical rather than execution order.  For X-VLA, reconstruct the
        # independently specified denoising chronology: the action outputs are
        # downstream of all ten flow steps and cannot be the first boundary.
        if path.parents[2].name == "xvla":
            chronological_labels = [
                label
                for step in range(10)
                for label in (
                    f"flow.step_{step:02d}.noisy_action",
                    f"flow.step_{step:02d}.prediction",
                )
                if label in flow_steps
            ]
            chronological_labels.extend(
                label for label in (
                    "actions.pre_libero_postprocess",
                    "actions.libero_postprocessed",
                ) if label in flow_steps
            )
            chronological_labels.extend(
                label for label in flow_steps if label not in chronological_labels
            )
        elif path.parents[2].name == "univla_baai":
            length = univla_flow_length(flow_steps)
            chronological_labels = [
                f"flow.step_{step:03d}.scores" for step in range(length or 0)
            ]
            chronological_labels.extend(
                label for label in flow_steps if label not in chronological_labels
            )
        else:
            chronological_labels = list(flow_steps)
        divergent = next(
            (label for label in chronological_labels if not flow_steps[label].get("exact")),
            None,
        )
        expected_boundary = divergent or ("returned_output" if not output.get("exact") else None)
        reported = item.get("first_divergence")
        if ((expected_boundary is None and reported is not None)
                or (expected_boundary is not None and (
                    not isinstance(reported, dict)
                    or reported.get("observed_boundary") != expected_boundary
                    or reported.get("metrics") != (flow_steps[divergent] if divergent else output)
                ))):
            issue(issues, f"ablation: first divergence differs from raw steps for {mode}: {request_id}")
            break
        if path.parents[2].name == "xvla":
            required_xvla_steps = {
                f"flow.step_{step:02d}.prediction" for step in range(10)
            }
            companion_index = item.get("companion_replay_index")
            if (not required_xvla_steps.issubset(flow_steps)
                    or item.get("batch_size") != 2
                    or not row.get("companion_replay_indices")
                    or companion_index != row["companion_replay_indices"][0]
                    or not isinstance(companion_index, int)
                    or manifest.get("observations", [])[companion_index].get("episode_id") == row.get("episode_id")):
                issue(issues, f"ablation: X-VLA step/companion schedule differs for {mode}: {request_id}")
                break
        if path.parents[2].name == "uva":
            required_uva_steps = {
                f"action.diffusion.step_{step:03d}" for step in range(100)
            }
            if not required_uva_steps.issubset(flow_steps):
                issue(issues, f"ablation: UVA action diffusion steps incomplete for {mode}: {request_id}")
                break
        if path.parents[2].name == "uva_joint":
            required_uva_steps = {
                f"{series}.diffusion.step_{step:03d}"
                for series in ("action", "video") for step in range(100)
            }
            if not required_uva_steps.issubset(flow_steps):
                issue(issues, f"ablation: UVA joint action/video steps incomplete for {mode}: {request_id}")
                break
        if path.parents[2].name == "rdt_1b":
            required_rdt_steps = {
                f"diffusion.{step}.{kind}"
                for step in range(5) for kind in ("prediction", "sample")
            }
            if not required_rdt_steps.issubset(flow_steps):
                issue(issues, f"ablation: RDT diffusion steps incomplete for {mode}: {request_id}")
                break
        if path.parents[2].name == "cogact_small":
            required_cogact_steps = {f"diffusion.{step}.sample" for step in range(10)}
            if not required_cogact_steps.issubset(flow_steps):
                issue(issues, f"ablation: CogACT DDIM steps incomplete for {mode}: {request_id}")
                break
        if path.parents[2].name == "vitra_vla_3b":
            required_vitra_steps = {f"ddim.step_{step:02d}.state" for step in range(10)}
            if not required_vitra_steps.issubset(flow_steps):
                issue(issues, f"ablation: VITRA DDIM steps incomplete for {mode}: {request_id}")
                break
        if path.parents[2].name == "openhelix":
            required_openhelix_steps = {
                f"flow.step_{step:02d}.state" for step in range(25)
            }
            if not required_openhelix_steps.issubset(flow_steps):
                issue(issues, f"ablation: OpenHelix DDPM steps incomplete for {mode}: {request_id}")
                break
        counts = item.get("override_invocations") or {}
        inapplicable_operations = item.get("inapplicable_operations") or {}
        if path.parents[2].name in ("uva", "uva_joint", "openhelix"):
            raw_sdpa_evidence = inapplicable_operations.get("explicit_sdpa")
            sdpa_evidence = raw_sdpa_evidence if isinstance(raw_sdpa_evidence, dict) else {}
            evidence_path = Path(sdpa_evidence.get("evidence_path", ""))
            if mode == "explicit_per_matrix_reference":
                if (set(inapplicable_operations) != {"explicit_sdpa"}
                        or not sdpa_evidence.get("reason")
                        or not evidence_path.is_file()
                        or sha256(evidence_path) != sdpa_evidence.get("evidence_sha256")
                        or "torch.bmm" not in evidence_path.read_text()
                        or "scaled_dot_product_attention" in evidence_path.read_text()
                        or not counts.get("explicit_bmm", 0)):
                    issue(issues, f"ablation: explicit-attention N/A contract differs for {mode}: {request_id}")
                    break
            elif inapplicable_operations:
                issue(issues, f"ablation: explicit-attention inapplicability attached to another mode: {mode}")
                break
        elif path.parents[2].name == "rdt_1b":
            allowed = {"convolution"} if mode in (
                "operators_plus_patch_projection", "explicit_per_matrix_reference"
            ) else set()
            if set(inapplicable_operations) != allowed:
                issue(issues, f"ablation: RDT operator inapplicability differs for {mode}: {request_id}")
                break
            if allowed:
                evidence = inapplicable_operations.get("convolution")
                evidence_path = Path(evidence.get("evidence_path", "")) if isinstance(evidence, dict) else Path("")
                trace = read_json(evidence_path) if evidence_path.is_file() else None
                if (not isinstance(evidence, dict) or not evidence.get("reason")
                        or not evidence_path.is_file()
                        or sha256(evidence_path) != evidence.get("evidence_sha256")
                        or not trace
                        or trace.get("adapter") != "scripts.model_invariance.adapters.rdt_maniskill"
                        or any(row.get("aten_operation") == "aten::convolution"
                               for row in trace.get("operations", []))
                        or counts.get("convolution", 0)):
                    issue(issues, f"ablation: RDT convolution N/A evidence differs for {mode}: {request_id}")
                    break
        elif inapplicable_operations:
            issue(issues, f"ablation: unexpected operator inapplicability for {mode}: {request_id}")
            break
        if ((mode == "native_deterministic") != (item.get("deterministic_algorithms") is True)
                or mode == "existing_invariant_ops" and not (
                    counts.get("mm", 0) or counts.get("addmm", 0) or counts.get("bmm", 0)
                )
                or mode in ("operators_plus_patch_projection", "explicit_per_matrix_reference")
                and "convolution" not in inapplicable_operations
                and not counts.get("convolution", 0)
                or mode == "explicit_per_matrix_reference"
                and "explicit_sdpa" not in inapplicable_operations
                and not counts.get("explicit_sdpa", 0)
                or mode == "full_invariant" and (
                    not output.get("exact") or any(not value.get("exact") for value in flow_steps.values())
                )):
            issue(issues, f"ablation: causal mode contract differs for {mode}: {request_id}")
            break
    for mode in modes:
        if mode not in inapplicable and records[mode] != expected_ids:
            issue(issues, f"ablation: {mode} has {len(records[mode])}/{len(expected_ids)} diagnostic requests")
    return {"complete": len(issues) == start, "modes": len(records) + len(inapplicable)}


def capacity_audit(path: Path, directory: Path, issues: list[str]) -> dict:
    start = len(issues)
    model_id = directory.parent.name
    required_sizes = capacity_batch_sizes(model_id)
    if not path.is_file():
        encoded = "/".join(f"B{size}" for size in required_sizes)
        issue(issues, f"capacity: missing {encoded} real-request probe {path}")
        return {"complete": False}
    manifest_path = directory / "replay_manifest.json"
    manifest = read_json(manifest_path) or {}
    observations = manifest.get("observations", [])
    manifest_hash = sha256(manifest_path) if manifest_path.is_file() else None
    by_request = {
        request_id: (row, noise_id)
        for row in observations if row.get("split") == "diagnostic"
        for noise_id, request_id in enumerate(row.get("request_ids", []))
    }
    required_configuration = EXPECTED_EXECUTION_CONFIG.get(model_id)
    grouped = defaultdict(dict)
    for item in stream_jsonl(path):
        size = item.get("batch_size")
        mode = item.get("implementation")
        kind = item.get("companion_type")
        position = item.get("target_position")
        if size not in required_sizes or mode not in POLICY_MODES or kind not in ("duplicate", "diverse") or position not in ("first", "middle", "last"):
            issue(issues, "capacity: unexpected size/mode/arrangement")
            break
        key = (kind, position)
        if key in grouped[(size, mode)]:
            issue(issues, f"capacity: duplicate B{size} {mode} {key}")
            break
        grouped[(size, mode)][key] = item
        source = by_request.get(item.get("request_id"))
        if item.get("replay_manifest_sha256") != manifest_hash or source is None or not item.get("gpu") or not item.get("torch"):
            issue(issues, f"capacity: unbound request/provenance for B{size} {mode} {key}")
            break
        if required_configuration is not None and item.get("adapter_execution_configuration") != required_configuration:
            issue(issues, f"capacity: unrepaired adapter execution configuration for B{size} {mode} {key}")
            break
        row, noise_id = source
        expected_ids = (
            [item["request_id"]] * (size - 1) if kind == "duplicate" else
            [observations[index]["request_ids"][noise_id] for index in row.get("companion_replay_indices", [])[: size - 1]]
        )
        if item.get("companion_ids") != expected_ids:
            issue(issues, f"capacity: companions differ from frozen real-request schedule for B{size} {mode} {key}")
            break
        status = item.get("status")
        if status == "ran":
            if not isinstance(item.get("peak_incremental_bytes"), int) or item["peak_incremental_bytes"] < 0 or not isinstance(item.get("output", {}).get("exact"), bool):
                issue(issues, f"capacity: missing measured output/memory for B{size} {mode} {key}")
                break
            if model_id == "pi_fast" and not {
                "output.actions", "output.official_decode_valid",
            }.issubset((item.get("output", {}).get("tensors") or {}).keys()):
                issue(issues, f"capacity: PI0-FAST omits official decode validity for B{size} {mode} {key}")
                break
            if mode == "batch_invariant" and not item["output"]["exact"]:
                issue(issues, f"capacity: invariant B{size} diverged for {key}")
        elif status == "cuda_oom":
            if "out of memory" not in str(item.get("error", "")).lower() or not isinstance(item.get("free_bytes_after_failure"), int):
                issue(issues, f"capacity: OOM lacks captured CUDA failure evidence for B{size} {mode} {key}")
        else:
            issue(issues, f"capacity: unknown status for B{size} {mode} {key}")
    wanted = {(kind, position) for kind in ("duplicate", "diverse") for position in ("first", "middle", "last")}
    feasible = {}
    for size in required_sizes:
        for mode in POLICY_MODES:
            rows = grouped.get((size, mode), {})
            oom = any(item.get("status") == "cuda_oom" for item in rows.values())
            if not rows or (not oom and set(rows) != wanted):
                issue(issues, f"capacity: B{size} {mode} needs six measured arrangements or captured OOM")
            feasible[f"{mode}|B{size}"] = bool(rows) and not oom and set(rows) == wanted
    return {"complete": len(issues) == start, "records": sum(map(len, grouped.values())), "feasible": feasible}


def large_policy_timing_audit(directory: Path, capacity: dict, issues: list[str]) -> dict:
    start = len(issues)
    feasible = capacity.get("feasible", {})
    audited = {}
    for size in capacity_batch_sizes(directory.parent.name):
        for mode in POLICY_MODES:
            key = f"{mode}|B{size}"
            if not feasible.get(key):
                continue
            path = directory / "performance" / f"policy_{mode}_B{size}_timings.jsonl"
            audited[key] = timing_audit(
                path, "policy", issues,
                required_policy_batches=(size,), expected_policy_modes=(mode,),
            )
    if not capacity.get("complete"):
        issue(issues, "large policy timing: cannot classify feasible shapes before capacity probe passes")
    return {"complete": len(issues) == start, "groups": audited}


def large_serving_shapes(capacity: dict) -> list[tuple[int, int]]:
    feasible = capacity.get("feasible", {})
    required = []
    if feasible.get("native|B16"):
        required.append((8, 16))
    if all(feasible.get(key) for key in ("native|B16", "batch_invariant|B16", "native|B32")):
        required.append((16, 32))
    return required


def large_serving_timing_audit(directory: Path, capacity: dict, issues: list[str]) -> dict:
    start = len(issues)
    audited = {}
    for logical, fixed in large_serving_shapes(capacity):
        path = directory / "performance" / f"serving_fixed_B{fixed}_timings.jsonl"
        if path.is_file():
            environment = next((row for row in stream_jsonl(path) if row.get("record_type") == "environment"), None)
            if not environment or (environment.get("logical_batch_size"), environment.get("fixed_batch_size")) != (logical, fixed):
                issue(issues, f"large serving timing: expected logical B{logical}/fixed B{fixed} in {path}")
        audited[f"logical_B{logical}|fixed_B{fixed}"] = timing_audit(path, "serving", issues)
    if not capacity.get("complete"):
        issue(issues, "large serving timing: cannot classify feasible shapes before capacity probe passes")
    return {"complete": len(issues) == start, "groups": audited}


def large_serving_queue_audit(directory: Path, capacity: dict, issues: list[str]) -> dict:
    start = len(issues)
    audited = {}
    for logical, fixed in large_serving_shapes(capacity):
        raw = queue_audit(
            directory / "serving" / f"requests_fixed_B{fixed}.jsonl", issues,
            logical_batch_size=logical, fixed_batch_size=fixed,
            serving_timings_path=directory / "performance" / f"serving_fixed_B{fixed}_timings.jsonl",
        )
        summary = queue_summary_audit(
            directory / "serving" / f"summary_fixed_B{fixed}.json",
            raw.get("group_counts", {}), raw.get("group_metrics", {}), issues,
        )
        audited[f"logical_B{logical}|fixed_B{fixed}"] = {"raw": raw, "summary": summary}
    if not capacity.get("complete"):
        issue(issues, "large serving queue: cannot classify feasible shapes before capacity probe passes")
    return {"complete": len(issues) == start, "groups": audited}


def launch_audit(path: Path, issues: list[str]) -> dict:
    start = len(issues)
    model_id = path.parents[2].name
    policy_sizes = (1, 2, 4) if model_id == "dexvla" else (1, 2, 4, 8)
    logical_size, fixed_size = (2, 4) if model_id == "dexvla" else (4, 8)
    report = read_json(path)
    if not report:
        issue(issues, f"performance: missing kernel-launch profile {path}")
        return {"complete": False}
    required_configuration = EXPECTED_EXECUTION_CONFIG.get(model_id)
    if required_configuration is not None and report.get("adapter_execution_configuration") != required_configuration:
        issue(issues, "performance: kernel-launch profile uses another adapter execution configuration")
    selection = report.get("input_selection")
    if selection is None:
        issue(issues, "performance: kernel-launch profile lacks frozen diagnostic inputs")
    else:
        timing_input_selection_audit(selection, fixed_size, issues)
        for timing_name in ("policy_timings.jsonl", "serving_timings.jsonl"):
            timing_path = path.parent / timing_name
            if timing_path.is_file():
                environment = next((row for row in stream_jsonl(timing_path) if row.get("record_type") == "environment"), None)
                if not environment or environment.get("input_selection") != selection:
                    issue(issues, f"performance: kernel-launch inputs differ from {timing_name}")
    fixed_shape = report.get("fixed_shape_metadata") or {}
    if fixed_shape.get("physical_batch_size") != fixed_size:
        issue(issues, f"performance: kernel-launch fixed-padded physical shape is not B{fixed_size}")
    required_fixed_shape = EXPECTED_FIXED_LANGUAGE_SHAPE.get(model_id)
    if required_fixed_shape is not None and any(fixed_shape.get(key) != value for key, value in required_fixed_shape.items()):
        issue(issues, "performance: kernel-launch fixed-padded prompt shape is not canonical")
    serving_timing = path.parent / "serving_timings.jsonl"
    if serving_timing.is_file():
        serving_fixed = [row for row in stream_jsonl(serving_timing) if row.get("record_type") == "fixed_shape"]
        if len(serving_fixed) != 1 or serving_fixed[0].get("metadata") != fixed_shape:
            issue(issues, "performance: kernel-launch fixed-padded shape differs from serving timing")
    profiles = report.get("profiles", [])
    required = {
        ("policy", mode, size)
        for mode in POLICY_MODES for size in policy_sizes
    } | {("serving", mode, None) for mode in SERVING_MODES}
    seen = set()
    for profile in profiles:
        key = (profile.get("kind"), profile.get("configuration"), profile.get("batch_size"))
        if key in seen:
            issue(issues, f"performance: duplicate kernel profile {key}")
        seen.add(key)
        if not isinstance(profile.get("kernel_launch_count"), int) or profile["kernel_launch_count"] < 1:
            issue(issues, f"performance: missing CUDA kernel launches for {key}")
        if not profile.get("cuda_kernels"):
            issue(issues, f"performance: missing CUDA kernel identities for {key}")
        elif profile.get("kernel_launch_count") != sum(profile["cuda_kernels"].values()):
            issue(issues, f"performance: CUDA kernel-launch count disagrees with profile events for {key}")
        if not profile.get("aten_operators"):
            issue(issues, f"performance: dispatched ATen operators absent from profile for {key}")
    if not required.issubset(seen):
        issue(issues, f"performance: kernel-launch coverage {len(required & seen)}/{len(required)}")
    return {"complete": len(issues) == start, "profiles": len(profiles)}


def flow_summary_audit(path: Path, identifier: str, issues: list[str], capacity: dict | None = None) -> dict:
    start = len(issues)
    report = read_json(path)
    if not report:
        issue(issues, f"heldout: missing flow-step summary {path}")
        return {"complete": False}
    if identifier in (
        "smolvla", "xvla", "internvla_a15_standard", "internvla_a15_optimized",
        "groot_n17", "wsa_base", "uva", "cogact_small", "dexvla",
        "lingbot_vla2", "xiaomi_robotics_1", "vitra_vla_3b", "openhelix",
    ):
        step_count = (
            4 if identifier == "groot_n17"
            else 100 if identifier == "uva"
            else 5 if identifier == "xiaomi_robotics_1"
            else 25 if identifier == "openhelix"
            else 10
        )
        if identifier == "cogact_small" and (
            report.get("maximum_steps") != 10
            or report.get("variable_length") is not False
            or report.get("flow_label_template") != "diffusion.{step}.sample"
        ):
            issue(issues, "heldout: CogACT flow summary does not use the fixed ten-step DDIM trace")
        if identifier == "vitra_vla_3b" and (
            report.get("maximum_steps") != 10
            or report.get("variable_length") is not False
            or report.get("flow_label_template") != "ddim.step_{step:02d}.state"
        ):
            issue(issues, "heldout: VITRA flow summary does not use the fixed ten-step DDIM trace")
        if identifier == "openhelix" and (
            report.get("maximum_steps") != 25
            or report.get("variable_length") is not False
            or report.get("flow_label_template") != "flow.step_{step:02d}.state"
        ):
            issue(issues, "heldout: OpenHelix flow summary does not use the fixed 25-step DDPM trace")
        implementations = report.get("implementations", {})
        if set(implementations) != set(POLICY_MODES):
            issue(issues, f"heldout: {identifier} flow summary lacks both implementations")
        for mode in POLICY_MODES:
            steps = implementations.get(mode, {})
            if set(steps) != {str(index) for index in range(step_count)}:
                issue(issues, f"heldout: {identifier} {mode} flow summary lacks {step_count} steps")
                continue
            capacity_value = capacity or {}
            request_count = expected_replay_counts(identifier)[2] * 3
            # B2's partition executes physical B1 and is omitted by the flow
            # summarizer; every other non-singleton arrangement contributes.
            sizes = replay_batch_sizes(identifier, capacity_value, mode)
            comparisons_per_request = sum(
                (5 if size == 2 else 8) for size in sizes if size > 1
            )
            expected_comparisons = request_count * comparisons_per_request
            for step, data in steps.items():
                if data.get("comparisons") != expected_comparisons:
                    issue(issues, f"heldout: {identifier} {mode} step {step} lacks {expected_comparisons} nonsingleton comparisons")
                if mode == "batch_invariant" and (
                    data.get("exact_count") != expected_comparisons or data.get("maximum_absolute_error") != 0.0
                ):
                    issue(issues, f"heldout: {identifier} invariant flow step {step} is not exact")
    elif identifier == "uva_joint":
        if (report.get("maximum_steps") != 100
                or report.get("variable_length") is not False
                or set(report.get("series", {})) != {"action", "video"}):
            issue(issues, "heldout: UVA joint flow summary lacks both fixed 100-step series")
            return {"complete": False}
        feasible = (capacity or {}).get("feasible", {})
        for series, template in (
            ("action", "action.diffusion.step_{step:03d}"),
            ("video", "video.diffusion.step_{step:03d}"),
        ):
            series_report = report["series"][series]
            if series_report.get("flow_label_template") != template:
                issue(issues, f"heldout: UVA joint {series} flow template differs")
                continue
            implementations = series_report.get("implementations", {})
            if set(implementations) != set(POLICY_MODES):
                issue(issues, f"heldout: UVA joint {series} summary lacks both implementations")
            for mode in POLICY_MODES:
                steps = implementations.get(mode, {})
                if set(steps) != {str(index) for index in range(100)}:
                    issue(issues, f"heldout: UVA joint {series} {mode} summary lacks 100 steps")
                    continue
                expected_comparisons = 2_400 * (21 + 8 * sum(
                    bool(feasible.get(f"{mode}|B{size}")) for size in (16, 32)
                ))
                for step, data in steps.items():
                    if data.get("comparisons") != expected_comparisons:
                        issue(issues, f"heldout: UVA joint {series} {mode} step {step} lacks comparisons")
                    if mode == "batch_invariant" and (
                        data.get("exact_count") != expected_comparisons
                        or data.get("maximum_absolute_error") != 0.0
                    ):
                        issue(issues, f"heldout: UVA joint invariant {series} step {step} is not exact")
    elif identifier == "rdt_1b":
        if (report.get("maximum_steps") != 5
                or report.get("variable_length") is not False
                or set(report.get("series", {})) != {"prediction", "sample"}):
            issue(issues, "heldout: RDT flow summary lacks both fixed five-step series")
            return {"complete": False}
        feasible = (capacity or {}).get("feasible", {})
        for series, template in (
            ("prediction", "diffusion.{step}.prediction"),
            ("sample", "diffusion.{step}.sample"),
        ):
            series_report = report["series"][series]
            if series_report.get("flow_label_template") != template:
                issue(issues, f"heldout: RDT {series} flow template differs")
                continue
            implementations = series_report.get("implementations", {})
            if set(implementations) != set(POLICY_MODES):
                issue(issues, f"heldout: RDT {series} summary lacks both implementations")
            for mode in POLICY_MODES:
                steps = implementations.get(mode, {})
                if set(steps) != {str(index) for index in range(5)}:
                    issue(issues, f"heldout: RDT {series} {mode} summary lacks five steps")
                    continue
                expected_comparisons = 2_400 * (21 + 8 * sum(
                    bool(feasible.get(f"{mode}|B{size}")) for size in (16, 32)
                ))
                for step, data in steps.items():
                    if data.get("comparisons") != expected_comparisons:
                        issue(issues, f"heldout: RDT {series} {mode} step {step} lacks comparisons")
                    if mode == "batch_invariant" and (
                        data.get("exact_count") != expected_comparisons
                        or data.get("maximum_absolute_error") != 0.0
                    ):
                        issue(issues, f"heldout: RDT invariant {series} step {step} is not exact")
    elif identifier in ("pi_fast", "spatialvla", "univla_baai"):
        model_name = {
            "pi_fast": "PI0-FAST", "spatialvla": "SpatialVLA",
            "univla_baai": "UniVLA",
        }[identifier]
        template = {
            "pi_fast": "flow.step_{step:03d}.prediction",
            "spatialvla": "generation.step_{step:03d}.scores",
            "univla_baai": "flow.step_{step:03d}.scores",
        }[identifier]
        length_function = {
            "pi_fast": pi_fast_flow_length,
            "spatialvla": spatialvla_flow_length,
            "univla_baai": univla_flow_length,
        }[identifier]
        maximum_steps = 80 if identifier == "univla_baai" else 256
        raw_path = path.parent / "invariance.jsonl.gz"
        if not raw_path.is_file():
            raw_path = path.parent / "invariance.jsonl"
        if (not raw_path.is_file()
                or report.get("variable_length") is not True
                or report.get("maximum_steps") != maximum_steps
                or report.get("flow_label_template") != template):
            issue(issues, f"heldout: {model_name} variable autoregressive summary provenance differs")
            return {"complete": False}
        try:
            source_matches = Path(report.get("source", "")).resolve() == raw_path.resolve()
        except (OSError, TypeError):
            source_matches = False
        if not source_matches:
            issue(issues, f"heldout: {model_name} flow summary points to another raw archive")
        raw_counts = Counter()
        raw_exact = Counter()
        raw_maximum = defaultdict(float)
        for item in stream_jsonl(raw_path):
            if item.get("batch_size", 0) <= 1:
                continue
            mode = item.get("implementation")
            length = length_function(item.get("flow_steps"))
            if mode not in POLICY_MODES or length is None:
                issue(issues, f"heldout: {model_name} raw flow summary input is malformed")
                break
            for step in range(length):
                metrics = item["flow_steps"][template.format(step=step)]
                key = (mode, step)
                raw_counts[key] += 1
                raw_exact[key] += int(metrics["exact"])
                raw_maximum[key] = max(raw_maximum[key], metrics["max_absolute_error"])
        implementations = report.get("implementations", {})
        if set(implementations) != set(POLICY_MODES):
            issue(issues, f"heldout: {model_name} flow summary lacks both implementations")
        for mode in POLICY_MODES:
            expected_steps = sorted(step for observed_mode, step in raw_counts if observed_mode == mode)
            if not expected_steps or expected_steps != list(range(expected_steps[-1] + 1)):
                issue(issues, f"heldout: {model_name} {mode} raw autoregressive steps are not contiguous")
                continue
            summary_steps = implementations.get(mode, {})
            if set(summary_steps) != {str(step) for step in expected_steps}:
                issue(issues, f"heldout: {model_name} {mode} summary omits observed autoregressive steps")
                continue
            for step in expected_steps:
                data = summary_steps[str(step)]
                key = (mode, step)
                if (data.get("comparisons") != raw_counts[key]
                        or data.get("exact_count") != raw_exact[key]
                        or data.get("maximum_absolute_error") != raw_maximum[key]):
                    issue(issues, f"heldout: {model_name} {mode} step {step} summary differs from raw evidence")
                    break
                if mode == "batch_invariant" and (
                    raw_exact[key] != raw_counts[key] or raw_maximum[key] != 0.0
                ):
                    issue(issues, f"heldout: {model_name} invariant autoregressive step {step} is not exact")
                    break
    elif report.get("status") == "not_applicable":
        evidence = report.get("evidence_path")
        evidence_path = Path(evidence) if isinstance(evidence, str) else None
        raw_path = path.parent / "invariance.jsonl.gz"
        if not raw_path.is_file():
            raw_path = path.parent / "invariance.jsonl"
        try:
            source_matches = Path(report.get("source", "")).resolve() == raw_path.resolve()
        except (OSError, TypeError):
            source_matches = False
        if (report.get("model_id") != identifier
                or not evidence_path or not evidence_path.is_file()
                or report.get("evidence_sha256") != sha256(evidence_path)
                or not report.get("reason") or not raw_path.is_file()
                or not source_matches or report.get("source_sha256") != sha256(raw_path)
                or report.get("implementations") != list(POLICY_MODES)):
            issue(issues, f"heldout: noniterative flow N/A lacks evidence for {identifier}")
        else:
            nonsingleton = 0
            observed_boundaries = None
            for item in stream_jsonl(raw_path):
                if item.get("batch_size", 0) <= 1:
                    continue
                nonsingleton += 1
                boundaries = item.get("flow_steps")
                if (item.get("implementation") not in POLICY_MODES
                        or not isinstance(boundaries, dict) or not boundaries
                        or any(label.startswith("flow.step_") for label in boundaries)):
                    issue(issues, f"heldout: {identifier} raw evidence is not a noniterative boundary trace")
                    break
                labels = set(boundaries)
                if observed_boundaries is None:
                    observed_boundaries = labels
                elif labels != observed_boundaries:
                    issue(issues, f"heldout: {identifier} noniterative boundary coverage varies")
                    break
            if (report.get("nonsingleton_records") != nonsingleton
                    or report.get("observed_boundaries") != sorted(observed_boundaries or ())):
                issue(issues, f"heldout: {identifier} noniterative N/A summary differs from raw evidence")
    elif not report.get("implementations"):
        issue(issues, f"heldout: missing iterative flow analysis for {identifier}")
    return {"complete": len(issues) == start}


def heldout_summary_audit(path: Path, raw: dict, issues: list[str]) -> dict:
    start = len(issues)
    report = read_json(path)
    if not report:
        issue(issues, f"heldout: missing numerical summary {path}")
        return {"complete": False}
    if report.get("fixed_request_noise_pairs") != raw.get("fidelity"):
        issue(issues, "heldout: numerical summary request count differs from raw fidelity")
    implementations = report.get("implementations", {})
    if set(implementations) != set(POLICY_MODES):
        issue(issues, "heldout: numerical summary lacks both implementations")
    for mode in POLICY_MODES:
        item = implementations.get(mode, {})
        if item.get("nonsingleton_arrangements") != raw.get("nonsingleton", {}).get(mode, 0):
            issue(issues, f"heldout: {mode} summary arrangement count differs from raw")
        if item.get("arrangement_violations") != raw.get("violations", {}).get(mode, 0):
            issue(issues, f"heldout: {mode} summary violation count differs from raw")
        if item.get("request_violations") != raw.get("request_violations", {}).get(mode, 0):
            issue(issues, f"heldout: {mode} summary request-violation count differs from raw")
        if any(field not in item for field in (
            "median_max_absolute_error", "p95_max_absolute_error", "p99_max_absolute_error",
            "maximum_absolute_error", "arrangement_violation_rate", "request_violation_rate",
        )):
            issue(issues, f"heldout: {mode} summary lacks required error statistics")
    return {"complete": len(issues) == start}


def uva_evaluator_evidence_audit(
    record: dict, identifier: str, checkpoint_weight: Path, manifest_sources: dict,
    conditions: set[str], expected_tasks: set[int], expected_pairs: set[tuple[int, int]],
    groups: dict, start_seed: int | None, issues: list[str],
) -> None:
    """Independently reconstruct UVA's exact action or joint B1/B4 campaign."""
    if identifier not in ("uva", "uva_joint"):
        issue(issues, f"closed loop: unsupported UVA identifier {identifier}")
        return
    if set(manifest_sources) != conditions:
        issue(issues, f"closed loop: {identifier} lacks four pinned evaluator condition manifests")

    task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
    task_manifest = read_json(task_manifest_path) or {}
    expected_checkpoint = Path("/workspace/assets/uva/libero10.ckpt")
    if (task_manifest.get("schema_version") != 1
            or task_manifest.get("task_suite") != "libero_10"
            or task_manifest.get("task_ids") != list(range(10))
            or task_manifest.get("task_descriptions") != UVA_TASK_DESCRIPTIONS
            or task_manifest.get("initializations_per_task") != 50
            or task_manifest.get("checkpoint_weight_path") != str(expected_checkpoint.resolve())
            or task_manifest.get("checkpoint_weight_sha256") != UVA_LIBERO10_CHECKPOINT_SHA256
            or task_manifest.get("checkpoint_repository_revision") != UVA_REPOSITORY_COMMIT
            or expected_tasks != set(range(10))):
        issue(issues, f"closed loop: {identifier} exact task/checkpoint manifest differs")
    if (checkpoint_weight != expected_checkpoint
            or not checkpoint_weight.is_file()
            or sha256(checkpoint_weight) != UVA_LIBERO10_CHECKPOINT_SHA256):
        issue(issues, f"closed loop: {identifier} exact checkpoint differs")

    wrapper = Path(__file__).with_name("run_uva_paired_libero.py")
    adapter = Path(__file__).parent / "model_invariance/adapters/uva_libero10.py"
    wrapper_sha = sha256(wrapper) if wrapper.is_file() else None
    adapter_sha = sha256(adapter) if adapter.is_file() else None
    evaluator_source = Path(record.get("official_evaluator_source_path", ""))
    expected_evaluator = Path("/workspace/models/uva/eval_sim.py")
    policy_source = Path(
        "/workspace/models/uva/unified_video_action/policy/unified_video_action_policy.py"
    )
    libero_runner = Path(
        "/workspace/models/uva/unified_video_action/env_runner/libero_image_runner.py"
    )
    libero_environment = Path(
        "/workspace/models/uva/unified_video_action/env/libero/envs/env_wrapper.py"
    )
    rotation_conversions = Path(
        "/workspace/assets/uva/pytorch3d_compat/pytorch3d/transforms/rotation_conversions.py"
    )
    pinned_sources = (
        (evaluator_source, expected_evaluator, UVA_EVALUATOR_SHA256, "official evaluator"),
        (policy_source, policy_source, UVA_POLICY_SHA256, "official policy"),
        (libero_runner, libero_runner, UVA_LIBERO_RUNNER_SHA256, "official LIBERO runner"),
        (
            libero_environment, libero_environment, UVA_LIBERO_ENVIRONMENT_SHA256,
            "official LIBERO environment",
        ),
        (
            rotation_conversions, rotation_conversions, UVA_ROTATION_CONVERSIONS_SHA256,
            "rotation conversion",
        ),
    )
    for observed, expected, digest, label in pinned_sources:
        if observed != expected or not observed.is_file() or sha256(observed) != digest:
            issue(issues, f"closed loop: {identifier} {label} differs")

    expected_path_kind = "joint" if identifier == "uva_joint" else "action"
    for condition in conditions & set(manifest_sources):
        item = manifest_sources[condition]
        manifest_path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
        if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
            issue(issues, f"closed loop: {identifier} condition manifest missing or hash-changed for {condition}")
            continue
        manifest = read_json(manifest_path) or {}
        required_batch = 1 if condition.endswith("singleton") else 4
        if (manifest.get("schema_version") != 1
                or manifest.get("model") != identifier
                or manifest.get("path") != expected_path_kind
                or manifest.get("condition") != condition
                or manifest.get("checkpoint") != str(expected_checkpoint)
                or manifest.get("checkpoint_sha256") != UVA_LIBERO10_CHECKPOINT_SHA256
                or manifest.get("official_repo_commit") != UVA_REPOSITORY_COMMIT
                or manifest.get("official_evaluator_sha256") != UVA_EVALUATOR_SHA256
                or manifest.get("official_policy_sha256") != UVA_POLICY_SHA256
                or manifest.get("official_libero_runner_sha256") != UVA_LIBERO_RUNNER_SHA256
                or manifest.get("official_libero_environment_sha256")
                != UVA_LIBERO_ENVIRONMENT_SHA256
                or manifest.get("rotation_conversions_sha256")
                != UVA_ROTATION_CONVERSIONS_SHA256
                or manifest.get("paired_adapter_sha256") != adapter_sha
                or manifest.get("paired_wrapper_sha256") != wrapper_sha
                or manifest.get("dependency_versions") != UVA_PACKAGES
                or not manifest.get("python") or not manifest.get("torch")
                or manifest.get("episodes_per_task") != 50
                or manifest.get("task_ids") != list(range(10))
                or manifest.get("seed") != start_seed
                or manifest.get("batch_size") != required_batch
                or manifest.get("invariant_ops") != condition.startswith("patched")
                or manifest.get("task_suite") != "libero_10"
                or manifest.get("episode_length") != 500
                or manifest.get("n_obs_steps") != 16
                or manifest.get("n_action_steps") != 8
                or manifest.get("control_mode") != "absolute"
                or manifest.get("explicit_init_state_by_initialization_id") is not True
                or manifest.get("request_local_noise") is not True
                or manifest.get("genuine_batched_policy_call") is not True
                or manifest.get("evidence_tier") != "full_paired_condition"):
            issue(issues, f"closed loop: {identifier} paired provenance differs for {condition}")

        paths = rollout_json_paths(manifest_path.parent / "rollouts")
        expected_file_count = 500 if required_batch == 1 else 130
        if len(paths) != expected_file_count:
            issue(
                issues,
                f"closed loop: {identifier} {condition} has "
                f"{len(paths)}/{expected_file_count} physical rollout groups",
            )
        expected_groups = {
            (task, tuple(range(begin, min(begin + required_batch, 50))))
            for task in expected_tasks for begin in range(0, 50, required_batch)
        }
        observed_groups, raw_rows = set(), {}
        for rollout_path in paths:
            payload = read_json(rollout_path) or {}
            task = payload.get("task_id")
            ids = payload.get("explicit_init_state_ids")
            seeds = payload.get("seeds")
            raw_episodes = payload.get("episodes")
            group_key = (task, tuple(ids) if isinstance(ids, list) else ())
            if (payload.get("task_group") != "libero_10"
                    or group_key not in expected_groups or group_key in observed_groups
                    or not isinstance(start_seed, int)
                    or seeds != [start_seed + value for value in group_key[1]]
                    or not isinstance(raw_episodes, list)
                    or len(raw_episodes) != len(group_key[1])):
                issue(issues, f"closed loop: {identifier} malformed physical rollout group {rollout_path}")
                continue
            observed_groups.add(group_key)
            for raw_episode in raw_episodes:
                initialization = raw_episode.get("initialization_id")
                raw_key = (task, initialization)
                sizes = raw_episode.get("policy_batch_sizes")
                inference_s = raw_episode.get("inference_s")
                video_hashes = raw_episode.get("prediction_video_sha256")
                expected_calls = len(sizes) if isinstance(sizes, list) else -1
                video_valid = (
                    isinstance(video_hashes, list)
                    and len(video_hashes) == expected_calls
                    and all(is_sha256_hex(value) for value in video_hashes)
                ) if identifier == "uva_joint" else "prediction_video_sha256" not in raw_episode
                if (raw_episode.get("task") != task
                        or raw_episode.get("task_group") != "libero_10"
                        or raw_episode.get("task_description") != UVA_TASK_DESCRIPTIONS[task]
                        or initialization not in group_key[1]
                        or raw_episode.get("reset_seed") != start_seed + initialization
                        or raw_key in raw_rows
                        or not isinstance(sizes, list) or not sizes
                        or sizes[0] != len(group_key[1])
                        or any(not isinstance(size, int) or not 1 <= size <= len(group_key[1])
                               for size in sizes)
                        or not isinstance(inference_s, list) or len(inference_s) != len(sizes)
                        or any(not isinstance(value, (int, float)) or value <= 0
                               for value in inference_s)
                        or not video_valid):
                    issue(issues, f"closed loop: {identifier} raw batched request differs in {rollout_path}")
                    continue
                raw_rows[raw_key] = (raw_episode, rollout_path)
        if observed_groups != expected_groups:
            issue(issues, f"closed loop: {identifier} {condition} physical group schedule is incomplete")
        if set(raw_rows) != expected_pairs:
            issue(issues, f"closed loop: {identifier} {condition} raw reset coverage differs")
        for raw_key in expected_pairs & set(raw_rows):
            combined = groups.get(raw_key, {}).get(condition)
            raw_episode, rollout_path = raw_rows[raw_key]
            if combined is None:
                continue
            expected_combined = dict(raw_episode)
            expected_combined.update({
                "condition": condition,
                "source_rollout_path": str(rollout_path.resolve()),
                "explicit_init_state_id": raw_key[1],
            })
            if combined != expected_combined:
                issue(
                    issues,
                    f"closed loop: {identifier} reduced episode differs from raw rollout "
                    f"for {condition} {raw_key}",
                )
                break


def official_success_row_matches_raw(
    row: dict, raw_successes: list[bool], *, allow_aggregate_only: bool,
) -> tuple[bool, str | None]:
    """Independently validate official per-episode or exact aggregate outcomes."""
    if (not isinstance(row, dict)
            or row.get("n_episodes") != len(raw_successes)
            or any(type(value) is not bool for value in raw_successes)
            or type(row.get("n_success")) is not int):
        return False, None
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return False, None
    if "successes" in metrics:
        successes = metrics["successes"]
        valid = (
            isinstance(successes, list)
            and len(successes) == len(raw_successes)
            and all(type(value) is bool for value in successes)
            and row["n_success"] == sum(successes)
            and successes == raw_successes
        )
        return valid, "per_episode" if valid else None
    valid = allow_aggregate_only and row["n_success"] == sum(raw_successes)
    return valid, "aggregate_only" if valid else None


def openvla_oft_evaluator_evidence_audit(
    record: dict, checkpoint_weight: Path, manifest_sources: dict, conditions: set[str],
    expected_tasks: set[int], expected_pairs: set[tuple[int, int]], groups: dict,
    start_seed: int | None, issues: list[str],
) -> None:
    """Independently reconstruct OpenVLA-OFT's exact paired physical campaign."""
    identifier = "openvla_oft"
    if set(manifest_sources) != conditions:
        issue(issues, f"closed loop: {identifier} lacks four pinned evaluator condition manifests")

    task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
    task_manifest = read_json(task_manifest_path) or {}
    checkpoint_parent = checkpoint_weight.parent
    expected_checkpoint_files = [
        {
            "path": str((checkpoint_parent / name).resolve()),
            "sha256": digest,
        }
        for name, digest in OPENVLA_OFT_CHECKPOINT_HASHES.items()
    ]
    if (task_manifest.get("schema_version") != 1
            or task_manifest.get("task_suite") != "libero_spatial"
            or task_manifest.get("task_ids") != list(range(10))
            or task_manifest.get("task_descriptions") != OPENVLA_OFT_TASK_DESCRIPTIONS
            or task_manifest.get("initializations_per_task") != 50
            or task_manifest.get("checkpoint_index_path") != str(checkpoint_weight.resolve())
            or task_manifest.get("checkpoint_index_sha256") != OPENVLA_OFT_INDEX_SHA256
            or task_manifest.get("checkpoint_files") != expected_checkpoint_files
            or record.get("exact_checkpoint_files") != expected_checkpoint_files
            or expected_tasks != set(range(10))):
        issue(issues, f"closed loop: {identifier} exact task/checkpoint manifest differs")
    if (checkpoint_weight.name != "model.safetensors.index.json"
            or not checkpoint_weight.is_file()
            or sha256(checkpoint_weight) != OPENVLA_OFT_INDEX_SHA256):
        issue(issues, f"closed loop: {identifier} exact checkpoint index differs")
    for item in expected_checkpoint_files:
        component = Path(item["path"])
        if not component.is_file() or sha256(component) != item["sha256"]:
            issue(issues, f"closed loop: {identifier} exact checkpoint component differs: {component.name}")

    wrapper_dir = Path(__file__).parent
    client_wrapper = wrapper_dir / "run_openvla_oft_paired_client.py"
    server_wrapper = wrapper_dir / "run_openvla_oft_paired_server.py"
    client_wrapper_sha = sha256(client_wrapper) if client_wrapper.is_file() else None
    server_wrapper_sha = sha256(server_wrapper) if server_wrapper.is_file() else None
    evaluator_source = Path(record.get("official_evaluator_source_path", ""))
    model_root = (
        evaluator_source.parents[3]
        if evaluator_source.is_file() and len(evaluator_source.parents) > 3
        else Path("")
    )
    expected_evaluator = model_root / "experiments/robot/libero/run_libero_eval.py"
    libero_utils = model_root / "experiments/robot/libero/libero_utils.py"
    policy_source = model_root / "prismatic/extern/hf/modeling_prismatic.py"
    libero_utils_sha = sha256(libero_utils) if libero_utils.is_file() else None
    policy_sha = sha256(policy_source) if policy_source.is_file() else None
    if evaluator_source != expected_evaluator or not evaluator_source.is_file():
        issue(issues, f"closed loop: {identifier} lacks pinned official evaluator implementation")

    deterministic_contract = {
        "inference_head": "l1_regression",
        "stochastic_request_inputs": "none",
        "do_sample": False,
    }
    deterministic_sha = hashlib.sha256(
        json.dumps(deterministic_contract, sort_keys=True).encode()
    ).hexdigest()
    task_manifest_sha = sha256(task_manifest_path) if task_manifest_path.is_file() else None

    for condition in conditions & set(manifest_sources):
        item = manifest_sources[condition]
        manifest_path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
        if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
            issue(issues, f"closed loop: {identifier} condition manifest missing or hash-changed for {condition}")
            continue
        manifest = read_json(manifest_path) or {}
        required_batch = 1 if condition.endswith("singleton") else 4
        required_padding = "canonical_right" if condition.startswith("patched") else "longest_in_batch_right"
        if (manifest.get("schema_version") != 1
                or manifest.get("model") != identifier
                or manifest.get("condition") != condition
                or manifest.get("batch_size") != required_batch
                or manifest.get("invariant_ops") != condition.startswith("patched")
                or manifest.get("episodes_per_task") != 50
                or manifest.get("seed") != start_seed
                or manifest.get("task_suite") != "libero_spatial"
                or manifest.get("episode_length") != 220
                or manifest.get("num_steps_wait") != 10
                or manifest.get("num_open_loop_steps") != 8
                or manifest.get("explicit_init_state_by_initialization_id") is not True
                or manifest.get("genuine_batched_policy_call") is not True
                or manifest.get("center_crop_area") != 0.9
                or manifest.get("num_images_in_input") != 2
                or manifest.get("use_proprio") is not True
                or manifest.get("prompt_padding") != required_padding
                or manifest.get("checkpoint") != str(checkpoint_parent.resolve())
                or manifest.get("checkpoint_index_sha256") != OPENVLA_OFT_INDEX_SHA256
                or manifest.get("checkpoint_files") != expected_checkpoint_files
                or manifest.get("official_evaluator_sha256") != record.get("official_evaluator_source_sha256")
                or manifest.get("official_libero_utils_sha256") != libero_utils_sha
                or manifest.get("official_policy_sha256") != policy_sha
                or manifest.get("paired_wrapper_sha256") != client_wrapper_sha
                or manifest.get("shared_rollout_wrapper_sha256") != server_wrapper_sha
                or manifest.get("task_manifest_path") != str(task_manifest_path.resolve())
                or manifest.get("task_manifest_sha256") != task_manifest_sha
                or manifest.get("deterministic_decoding_contract") != deterministic_contract
                or manifest.get("deterministic_decoding_sha256") != deterministic_sha):
            issue(issues, f"closed loop: {identifier} paired provenance differs for {condition}")

        server_metadata = read_json(manifest_path.parent / "server_metadata.json") or {}
        if (server_metadata.get("protocol_version") != "2.1"
                or server_metadata.get("policy_type") != identifier
                or server_metadata.get("checkpoint") != str(checkpoint_parent.resolve())
                or server_metadata.get("task_suite") != "libero_spatial"
                or server_metadata.get("paired_condition") != condition
                or server_metadata.get("genuine_batched_policy_call") is not True
                or server_metadata.get("maximum_request_batch") != required_batch
                or server_metadata.get("batch_invariant_ops") != condition.startswith("patched")
                or server_metadata.get("request_local_noise") is not False
                or server_metadata.get("deterministic_contract") != deterministic_contract
                or server_metadata.get("deterministic_contract_sha256") != deterministic_sha
                or server_metadata.get("prompt_padding") != required_padding
                or not isinstance(server_metadata.get("canonical_prompt_length"), int)
                or server_metadata.get("preprocessing_owner") != "server_official_policy_adapter"
                or server_metadata.get("center_crop_area") != 0.9
                or server_metadata.get("num_images_in_input") != 2
                or server_metadata.get("use_proprio") is not True
                or server_metadata.get("action_chunk_size") != 8
                or server_metadata.get("action_dim") != 7
                or server_metadata.get("checkpoint_files") != expected_checkpoint_files):
            issue(issues, f"closed loop: {identifier} server handshake differs for {condition}")

        paths = rollout_json_paths(manifest_path.parent / "rollouts")
        expected_file_count = 500 if required_batch == 1 else 130
        if len(paths) != expected_file_count:
            issue(
                issues,
                f"closed loop: {identifier} {condition} has "
                f"{len(paths)}/{expected_file_count} physical rollout groups",
            )
        expected_groups = {
            (task, tuple(range(start, min(start + required_batch, 50))))
            for task in expected_tasks for start in range(0, 50, required_batch)
        }
        observed_groups = set()
        raw_rows = {}
        for rollout_path in paths:
            payload = read_json(rollout_path) or {}
            task = payload.get("task_id")
            seeds = payload.get("seeds")
            explicit_ids = payload.get("explicit_init_state_ids")
            raw_episodes = payload.get("episodes")
            group_key = (task, tuple(explicit_ids) if isinstance(explicit_ids, list) else ())
            if (payload.get("task_group") != "libero_spatial"
                    or group_key not in expected_groups or group_key in observed_groups
                    or not isinstance(seeds, list) or not isinstance(raw_episodes, list)
                    or len(seeds) != len(group_key[1])
                    or len(raw_episodes) != len(group_key[1])
                    or not isinstance(start_seed, int)
                    or seeds != [start_seed + value for value in group_key[1]]):
                issue(issues, f"closed loop: {identifier} malformed physical rollout group {rollout_path}")
                continue
            observed_groups.add(group_key)
            for raw_episode in raw_episodes:
                initialization = raw_episode.get("initialization_id")
                raw_key = (task, initialization)
                sizes = raw_episode.get("policy_batch_sizes")
                if (raw_episode.get("task") != task
                        or raw_episode.get("task_group") != "libero_spatial"
                        or raw_episode.get("task_description") != OPENVLA_OFT_TASK_DESCRIPTIONS[task]
                        or initialization not in group_key[1]
                        or not isinstance(start_seed, int)
                        or raw_episode.get("reset_seed") != start_seed + initialization
                        or raw_episode.get("first_noise_sha256") != deterministic_sha
                        or raw_key in raw_rows
                        or not isinstance(sizes, list) or not sizes
                        or sizes[0] != len(group_key[1])
                        or any(not isinstance(size, int) or not 1 <= size <= len(group_key[1]) for size in sizes)):
                    issue(issues, f"closed loop: {identifier} raw batched request differs in {rollout_path}")
                    continue
                raw_rows[raw_key] = (raw_episode, rollout_path)
        if observed_groups != expected_groups:
            issue(issues, f"closed loop: {identifier} {condition} physical group schedule is incomplete")
        if set(raw_rows) != expected_pairs:
            issue(issues, f"closed loop: {identifier} {condition} raw reset coverage differs")
        for raw_key in expected_pairs & set(raw_rows):
            combined = groups.get(raw_key, {}).get(condition)
            raw_episode, rollout_path = raw_rows[raw_key]
            if combined is None:
                continue
            expected_combined = dict(raw_episode)
            expected_combined.update({
                "condition": condition,
                "source_rollout_path": str(rollout_path.resolve()),
                "explicit_init_state_id": raw_key[1],
            })
            if combined != expected_combined:
                issue(
                    issues,
                    f"closed loop: {identifier} reduced episode differs from raw rollout "
                    f"for {condition} {raw_key}",
                )
                break


def wsa_evaluator_evidence_audit(
    record: dict, checkpoint_weight: Path, manifest_sources: dict, conditions: set[str],
    expected_tasks: set[int], expected_pairs: set[tuple[int, int]], groups: dict,
    start_seed: int | None, issues: list[str],
) -> None:
    """Independently reconstruct WSA's exact stochastic B1/B4 LIBERO campaign."""
    identifier = "wsa_base"
    if set(manifest_sources) != conditions:
        issue(issues, f"closed loop: {identifier} lacks four pinned evaluator condition manifests")
    expected_artifacts = [
        {"path": str(Path(name).resolve()), "sha256": digest}
        for name, digest in WSA_ARTIFACT_HASHES.items()
    ]
    task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
    task_manifest = read_json(task_manifest_path) or {}
    if (task_manifest.get("schema_version") != 1
            or task_manifest.get("task_suite") != "libero_10"
            or task_manifest.get("task_ids") != list(range(10))
            or task_manifest.get("task_descriptions") != WSA_TASK_DESCRIPTIONS
            or task_manifest.get("initializations_per_task") != 50
            or task_manifest.get("artifact_files") != expected_artifacts
            or task_manifest.get("checkpoint_files") != expected_artifacts
            or record.get("exact_checkpoint_files") != expected_artifacts
            or expected_tasks != set(range(10))):
        issue(issues, f"closed loop: {identifier} exact task/checkpoint manifest differs")
    if (checkpoint_weight != Path("/workspace/assets/wsa/checkpoint/model.safetensors")
            or not checkpoint_weight.is_file()
            or sha256(checkpoint_weight) != WSA_CHECKPOINT_SHA256):
        issue(issues, f"closed loop: {identifier} exact checkpoint differs")
    for item in expected_artifacts:
        artifact = Path(item["path"])
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            issue(issues, f"closed loop: {identifier} exact artifact differs: {artifact.name}")

    wrapper_dir = Path(__file__).parent
    client_wrapper = wrapper_dir / "run_wsa_paired_client.py"
    server_wrapper = wrapper_dir / "run_wsa_paired_server.py"
    client_sha = sha256(client_wrapper) if client_wrapper.is_file() else None
    server_sha = sha256(server_wrapper) if server_wrapper.is_file() else None
    evaluator_source = Path(record.get("official_evaluator_source_path", ""))
    model_root = (
        evaluator_source.parents[2]
        if evaluator_source.is_file() and len(evaluator_source.parents) > 2
        else Path("")
    )
    expected_evaluator = model_root / "evaluation/Libero/inference.py"
    official_server = model_root / "evaluation/Libero/model_server.py"
    official_policy = model_root / "src/lerobot/policies/WSA_Base/modeling_wsa_base.py"
    official_server_sha = sha256(official_server) if official_server.is_file() else None
    official_policy_sha = sha256(official_policy) if official_policy.is_file() else None
    if evaluator_source != expected_evaluator or not evaluator_source.is_file():
        issue(issues, f"closed loop: {identifier} lacks pinned official evaluator implementation")
    task_manifest_sha = sha256(task_manifest_path) if task_manifest_path.is_file() else None

    for condition in conditions & set(manifest_sources):
        item = manifest_sources[condition]
        manifest_path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
        if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
            issue(issues, f"closed loop: {identifier} condition manifest missing or hash-changed for {condition}")
            continue
        manifest = read_json(manifest_path) or {}
        required_batch = 1 if condition.endswith("singleton") else 4
        if (manifest.get("schema_version") != 1
                or manifest.get("model") != identifier
                or manifest.get("condition") != condition
                or manifest.get("batch_size") != required_batch
                or manifest.get("invariant_ops") != condition.startswith("patched")
                or manifest.get("episodes_per_task") != 50
                or manifest.get("seed") != start_seed
                or manifest.get("task_suite") != "libero_10"
                or manifest.get("episode_length") != 520
                or manifest.get("num_steps_wait") != 10
                or manifest.get("num_open_loop_steps") != 10
                or manifest.get("image_history_interval") != 15
                or manifest.get("control_mode") != "absolute"
                or manifest.get("gripper_mode") != "libero_open_prob"
                or manifest.get("explicit_init_state_by_initialization_id") is not True
                or manifest.get("request_local_noise") is not True
                or manifest.get("noise_shape") != [10, 32]
                or manifest.get("flow_steps") != 10
                or manifest.get("genuine_batched_policy_call") is not True
                or manifest.get("checkpoint") != str(checkpoint_weight.parent.resolve())
                or manifest.get("checkpoint_sha256") != WSA_CHECKPOINT_SHA256
                or manifest.get("artifact_files") != expected_artifacts
                or manifest.get("official_evaluator_sha256") != record.get("official_evaluator_source_sha256")
                or manifest.get("official_server_sha256") != official_server_sha
                or manifest.get("official_policy_sha256") != official_policy_sha
                or manifest.get("paired_wrapper_sha256") != client_sha
                or manifest.get("shared_rollout_wrapper_sha256") != server_sha
                or manifest.get("task_manifest_path") != str(task_manifest_path.resolve())
                or manifest.get("task_manifest_sha256") != task_manifest_sha):
            issue(issues, f"closed loop: {identifier} paired provenance differs for {condition}")

        metadata = read_json(manifest_path.parent / "server_metadata.json") or {}
        if (metadata.get("protocol_version") != "2.1"
                or metadata.get("policy_type") != identifier
                or metadata.get("checkpoint") != str(checkpoint_weight.parent.resolve())
                or metadata.get("task_suite") != "libero_10"
                or metadata.get("paired_condition") != condition
                or metadata.get("genuine_batched_policy_call") is not True
                or metadata.get("maximum_request_batch") != required_batch
                or metadata.get("batch_invariant_ops") != condition.startswith("patched")
                or metadata.get("request_local_noise") is not True
                or metadata.get("flow_steps") != 10
                or metadata.get("noise_shape") != [10, 32]
                or metadata.get("image_history_interval") != 15
                or metadata.get("num_images_in_input") != 3
                or metadata.get("third_image_masked") is not True
                or metadata.get("action_chunk_size") != 10
                or metadata.get("action_dim") != 7
                or metadata.get("control_mode") != "absolute"
                or metadata.get("gripper_mode") != "libero_open_prob"
                or metadata.get("preprocessing_owner") != "server_official_policy_adapter"
                or metadata.get("artifact_files") != expected_artifacts):
            issue(issues, f"closed loop: {identifier} server handshake differs for {condition}")

        paths = rollout_json_paths(manifest_path.parent / "rollouts")
        expected_file_count = 500 if required_batch == 1 else 130
        if len(paths) != expected_file_count:
            issue(
                issues,
                f"closed loop: {identifier} {condition} has "
                f"{len(paths)}/{expected_file_count} physical rollout groups",
            )
        expected_groups = {
            (task, tuple(range(begin, min(begin + required_batch, 50))))
            for task in expected_tasks for begin in range(0, 50, required_batch)
        }
        observed_groups, raw_rows = set(), {}
        for rollout_path in paths:
            payload = read_json(rollout_path) or {}
            task = payload.get("task_id")
            ids = payload.get("explicit_init_state_ids")
            seeds = payload.get("seeds")
            raw_episodes = payload.get("episodes")
            group_key = (task, tuple(ids) if isinstance(ids, list) else ())
            if (payload.get("task_group") != "libero_10"
                    or group_key not in expected_groups or group_key in observed_groups
                    or not isinstance(start_seed, int)
                    or seeds != [start_seed + value for value in group_key[1]]
                    or not isinstance(raw_episodes, list)
                    or len(raw_episodes) != len(group_key[1])):
                issue(issues, f"closed loop: {identifier} malformed physical rollout group {rollout_path}")
                continue
            observed_groups.add(group_key)
            for raw_episode in raw_episodes:
                initialization = raw_episode.get("initialization_id")
                raw_key = (task, initialization)
                sizes = raw_episode.get("policy_batch_sizes")
                if (raw_episode.get("task") != task
                        or raw_episode.get("task_group") != "libero_10"
                        or raw_episode.get("task_description") != WSA_TASK_DESCRIPTIONS[task]
                        or initialization not in group_key[1]
                        or raw_episode.get("reset_seed") != start_seed + initialization
                        or raw_key in raw_rows
                        or not isinstance(sizes, list) or not sizes
                        or sizes[0] != len(group_key[1])
                        or any(not isinstance(size, int) or not 1 <= size <= len(group_key[1]) for size in sizes)):
                    issue(issues, f"closed loop: {identifier} raw batched request differs in {rollout_path}")
                    continue
                raw_rows[raw_key] = (raw_episode, rollout_path)
        if observed_groups != expected_groups:
            issue(issues, f"closed loop: {identifier} {condition} physical group schedule is incomplete")
        if set(raw_rows) != expected_pairs:
            issue(issues, f"closed loop: {identifier} {condition} raw reset coverage differs")
        for raw_key in expected_pairs & set(raw_rows):
            combined = groups.get(raw_key, {}).get(condition)
            raw_episode, rollout_path = raw_rows[raw_key]
            if combined is None:
                continue
            expected_combined = dict(raw_episode)
            expected_combined.update({
                "condition": condition,
                "source_rollout_path": str(rollout_path.resolve()),
                "explicit_init_state_id": raw_key[1],
            })
            if combined != expected_combined:
                issue(
                    issues,
                    f"closed loop: {identifier} reduced episode differs from raw rollout "
                    f"for {condition} {raw_key}",
                )
                break


def rdt_evaluator_evidence_audit(
    record: dict, checkpoint_weight: Path, manifest_sources: dict, conditions: set[str],
    expected_tasks: set[int], expected_pairs: set[tuple[int, int]], groups: dict,
    start_seed: int | None, issues: list[str],
) -> None:
    """Independently reconstruct RDT's exact stochastic five-task campaign."""
    identifier = "rdt_1b"
    if set(manifest_sources) != conditions:
        issue(issues, f"closed loop: {identifier} lacks four pinned evaluator condition manifests")
    expected_artifacts = [
        {"path": str(Path(name).resolve()), "sha256": digest}
        for name, digest in RDT_ARTIFACT_HASHES.items()
    ]
    task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
    task_manifest = read_json(task_manifest_path) or {}
    if (task_manifest.get("schema_version") != 1
            or task_manifest.get("task_suite") != "maniskill_five"
            or task_manifest.get("task_ids") != list(range(5))
            or task_manifest.get("task_names") != RDT_TASK_NAMES
            or task_manifest.get("initializations_per_task") != 50
            or task_manifest.get("base_seed") != 20241201
            or task_manifest.get("checkpoint_repository")
            != "robotics-diffusion-transformer/maniskill-model"
            or task_manifest.get("checkpoint_repository_revision")
            != RDT_CHECKPOINT_REPOSITORY_REVISION
            or task_manifest.get("artifact_files") != expected_artifacts
            or task_manifest.get("checkpoint_files") != expected_artifacts
            or record.get("exact_checkpoint_files") != expected_artifacts
            or expected_tasks != set(range(5))):
        issue(issues, f"closed loop: {identifier} exact task/checkpoint manifest differs")
    exact_checkpoint = Path("/workspace/assets/rdt/checkpoint/rdt/mp_rank_00_model_states.pt")
    if (checkpoint_weight != exact_checkpoint or not checkpoint_weight.is_file()
            or sha256(checkpoint_weight) != RDT_ARTIFACT_HASHES[str(exact_checkpoint)]):
        issue(issues, f"closed loop: {identifier} exact checkpoint differs")
    for item in expected_artifacts:
        artifact = Path(item["path"])
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            issue(issues, f"closed loop: {identifier} exact artifact differs: {artifact.name}")

    wrapper_dir = Path(__file__).parent
    client_wrapper = wrapper_dir / "run_rdt_paired_client.py"
    server_wrapper = wrapper_dir / "run_rdt_paired_server.py"
    client_sha = sha256(client_wrapper) if client_wrapper.is_file() else None
    server_sha = sha256(server_wrapper) if server_wrapper.is_file() else None
    evaluator_source = Path(record.get("official_evaluator_source_path", ""))
    model_root = (
        evaluator_source.parents[1]
        if evaluator_source.is_file() and len(evaluator_source.parents) > 1
        else Path("")
    )
    expected_evaluator = model_root / "eval_sim/eval_rdt_maniskill.py"
    official_wrapper = model_root / "scripts/maniskill_model.py"
    official_policy = model_root / "models/rdt_runner.py"
    official_wrapper_sha = sha256(official_wrapper) if official_wrapper.is_file() else None
    official_policy_sha = sha256(official_policy) if official_policy.is_file() else None
    if evaluator_source != expected_evaluator or not evaluator_source.is_file():
        issue(issues, f"closed loop: {identifier} lacks pinned official evaluator implementation")
    task_manifest_sha = sha256(task_manifest_path) if task_manifest_path.is_file() else None

    for condition in conditions & set(manifest_sources):
        item = manifest_sources[condition]
        manifest_path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
        if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
            issue(issues, f"closed loop: {identifier} condition manifest missing or hash-changed for {condition}")
            continue
        manifest = read_json(manifest_path) or {}
        required_batch = 1 if condition.endswith("singleton") else 4
        required_padding = "fixed_right_30" if condition.startswith("patched") else "longest_in_batch_right"
        if (manifest.get("schema_version") != 1
                or manifest.get("model") != identifier or manifest.get("condition") != condition
                or manifest.get("batch_size") != required_batch
                or manifest.get("invariant_ops") != condition.startswith("patched")
                or manifest.get("episodes_per_task") != 50 or manifest.get("seed") != start_seed
                or manifest.get("task_suite") != "maniskill_five"
                or manifest.get("episode_length") != 400
                or manifest.get("control_mode") != "pd_joint_pos"
                or manifest.get("control_frequency") != 25
                or manifest.get("action_chunk_size") != 64
                or manifest.get("executed_action_stride") != 4
                or manifest.get("request_local_noise") is not True
                or manifest.get("noise_shape") != [64, 128]
                or manifest.get("diffusion_steps") != 5
                or manifest.get("genuine_batched_policy_call") is not True
                or manifest.get("language_padding") != required_padding
                or manifest.get("explicit_init_state_by_initialization_id") is not True
                or manifest.get("checkpoint") != str(checkpoint_weight.parents[1].resolve())
                or manifest.get("checkpoint_sha256") != record.get("checkpoint_weight_sha256")
                or manifest.get("artifact_files") != expected_artifacts
                or manifest.get("official_evaluator_sha256") != record.get("official_evaluator_source_sha256")
                or manifest.get("official_wrapper_sha256") != official_wrapper_sha
                or manifest.get("official_policy_sha256") != official_policy_sha
                or manifest.get("paired_wrapper_sha256") != client_sha
                or manifest.get("shared_rollout_wrapper_sha256") != server_sha
                or manifest.get("task_manifest_path") != str(task_manifest_path.resolve())
                or manifest.get("task_manifest_sha256") != task_manifest_sha):
            issue(issues, f"closed loop: {identifier} paired provenance differs for {condition}")

        metadata = read_json(manifest_path.parent / "server_metadata.json") or {}
        if (metadata.get("protocol_version") != "2.1" or metadata.get("policy_type") != identifier
                or metadata.get("checkpoint") != str(checkpoint_weight.parents[1].resolve())
                or metadata.get("task_suite") != "maniskill_five"
                or metadata.get("paired_condition") != condition
                or metadata.get("genuine_batched_policy_call") is not True
                or metadata.get("maximum_request_batch") != required_batch
                or metadata.get("batch_invariant_ops") != condition.startswith("patched")
                or metadata.get("request_local_noise") is not True
                or metadata.get("diffusion_steps") != 5
                or metadata.get("noise_shape") != [64, 128]
                or metadata.get("language_padding") != required_padding
                or metadata.get("canonical_language_tokens") != 30
                or metadata.get("image_history") != 2 or metadata.get("num_cameras") != 3
                or metadata.get("action_chunk_size") != 64
                or metadata.get("executed_action_stride") != 4
                or metadata.get("action_dim") != 8
                or metadata.get("control_mode") != "pd_joint_pos"
                or metadata.get("control_frequency") != 25
                or metadata.get("preprocessing_owner") != "server_official_maniskill_adapter"
                or metadata.get("artifact_files") != expected_artifacts):
            issue(issues, f"closed loop: {identifier} server handshake differs for {condition}")

        paths = rollout_json_paths(manifest_path.parent / "rollouts")
        expected_file_count = 250 if required_batch == 1 else 65
        if len(paths) != expected_file_count:
            issue(issues, f"closed loop: {identifier} {condition} has "
                          f"{len(paths)}/{expected_file_count} physical rollout groups")
        expected_groups = {
            (task, tuple(range(begin, min(begin + required_batch, 50))))
            for task in expected_tasks for begin in range(0, 50, required_batch)
        }
        observed_groups, raw_rows = set(), {}
        for rollout_path in paths:
            payload = read_json(rollout_path) or {}
            task = payload.get("task_id")
            ids = payload.get("explicit_init_state_ids")
            seeds = payload.get("seeds")
            raw_episodes = payload.get("episodes")
            group_key = (task, tuple(ids) if isinstance(ids, list) else ())
            if (payload.get("task_group") != "maniskill_five"
                    or task not in expected_tasks or payload.get("task_name") != RDT_TASK_NAMES[task]
                    or group_key not in expected_groups or group_key in observed_groups
                    or not isinstance(start_seed, int)
                    or seeds != [start_seed + value for value in group_key[1]]
                    or not isinstance(raw_episodes, list)
                    or len(raw_episodes) != len(group_key[1])):
                issue(issues, f"closed loop: {identifier} malformed physical rollout group {rollout_path}")
                continue
            observed_groups.add(group_key)
            for raw_episode in raw_episodes:
                initialization = raw_episode.get("initialization_id")
                raw_key = (task, initialization)
                sizes = raw_episode.get("policy_batch_sizes")
                if (raw_episode.get("task") != task
                        or raw_episode.get("task_group") != "maniskill_five"
                        or raw_episode.get("task_name") != RDT_TASK_NAMES[task]
                        or initialization not in group_key[1]
                        or raw_episode.get("reset_seed") != start_seed + initialization
                        or raw_key in raw_rows or not isinstance(sizes, list) or not sizes
                        or sizes[0] != len(group_key[1])
                        or any(not isinstance(size, int) or not 1 <= size <= len(group_key[1]) for size in sizes)):
                    issue(issues, f"closed loop: {identifier} raw batched request differs in {rollout_path}")
                    continue
                raw_rows[raw_key] = (raw_episode, rollout_path)
        if observed_groups != expected_groups:
            issue(issues, f"closed loop: {identifier} {condition} physical group schedule is incomplete")
        if set(raw_rows) != expected_pairs:
            issue(issues, f"closed loop: {identifier} {condition} raw reset coverage differs")
        for raw_key in expected_pairs & set(raw_rows):
            combined = groups.get(raw_key, {}).get(condition)
            raw_episode, rollout_path = raw_rows[raw_key]
            if combined is None:
                continue
            expected_combined = dict(raw_episode)
            expected_combined.update({
                "condition": condition, "source_rollout_path": str(rollout_path.resolve()),
                "explicit_init_state_id": raw_key[1],
            })
            if combined != expected_combined:
                issue(issues, f"closed loop: {identifier} reduced episode differs from raw rollout "
                              f"for {condition} {raw_key}")
                break


def spatialvla_evaluator_evidence_audit(
    record: dict, episodes_path: Path, issues: list[str],
) -> None:
    """Independently reconstruct the non-uniform 12-task SimplerEnv campaign."""
    conditions = (
        "native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic",
    )
    condition_set = set(conditions)
    task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
    if (not task_manifest_path.is_file()
            or sha256(task_manifest_path) != record.get("evaluator_task_manifest_sha256")):
        issue(issues, "closed loop: spatialvla lacks a pinned evaluator task manifest")
        return
    task_manifest = read_json(task_manifest_path) or {}
    tasks = task_manifest.get("tasks") or []
    if (task_manifest.get("schema_version") != 1
            or task_manifest.get("model_id") != "spatialvla"
            or task_manifest.get("repository_revision") != "ccfe3809766839a2fcfb7a3d3c9abff585189188"
            or task_manifest.get("simulator_revision") != "cd45dd27dc6bb26d048cb6570cdab4e3f935cc37"
            or task_manifest.get("base_seed") != 20261005
            or task_manifest.get("task_count") != 12
            or task_manifest.get("initializations_per_condition") != 496
            or task_manifest.get("logical_episodes_all_conditions") != 1984
            or len(tasks) != 12
            or sum(task.get("initializations", 0) for task in tasks) != 496
            or [task.get("initializations") for task in tasks] != [24] * 4 + [50] * 8
            or [task.get("task_id") for task in tasks] != list(range(12))):
        issue(issues, "closed loop: spatialvla pinned 12-task schedule differs")
        return
    for source in task_manifest.get("source_files", []):
        source_path = Path(source.get("path", ""))
        if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
            issue(issues, f"closed loop: spatialvla evaluator source changed: {source_path}")
    expected_artifacts = [
        {"path": str((SPATIALVLA_CHECKPOINT / name).resolve()), "sha256": digest}
        for name, digest in SPATIALVLA_ARTIFACT_HASHES.items()
    ]
    if record.get("exact_checkpoint_files") != expected_artifacts:
        issue(issues, "closed loop: spatialvla exact checkpoint inventory differs")
    for item in expected_artifacts:
        artifact = Path(item["path"])
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            issue(issues, f"closed loop: spatialvla checkpoint artifact changed: {artifact}")
    checkpoint_weight = SPATIALVLA_CHECKPOINT / "model.safetensors.index.json"
    if (record.get("checkpoint_weight_path") != str(checkpoint_weight.resolve())
            or record.get("checkpoint_weight_sha256") != SPATIALVLA_ARTIFACT_HASHES[checkpoint_weight.name]):
        issue(issues, "closed loop: spatialvla checkpoint anchor differs")
    if record.get("paired_start_seed") != 20261005:
        issue(issues, "closed loop: spatialvla reset seed schedule differs")

    rows = list(stream_jsonl(episodes_path))
    expected_pairs = {
        (task["task_id"], initialization)
        for task in tasks for initialization in range(task["initializations"])
    }
    grouped: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row.get("task"), row.get("initialization_id"))
        condition = row.get("condition")
        action = row.get("action_sequence")
        raw = row.get("raw_action_sequence")
        chunks = row.get("raw_action_chunks")
        trajectory = row.get("trajectory")
        length = len(action) if isinstance(action, list) else -1
        if (key not in expected_pairs or condition not in condition_set
                or condition in grouped[key]
                or row.get("reset_seed") != 20261005 + key[1]
                or row.get("explicit_initialization_id") != key[1]
                or row.get("deterministic_policy") is not True
                or not isinstance(row.get("success"), bool)
                or not isinstance(row.get("initial_observation_sha256"), str)
                or len(row["initial_observation_sha256"]) != 64
                or length <= 0 or not isinstance(raw, list) or len(raw) != length
                or not isinstance(chunks, list) or len(chunks) != length
                or not isinstance(trajectory, list) or len(trajectory) != length + 1
                or len(row.get("rewards", [])) != length
                or len(row.get("terminated", [])) != length
                or len(row.get("truncated", [])) != length
                or len(row.get("policy_batch_sizes", [])) != length
                or row["success"] != any(row["terminated"])):
            issue(issues, f"closed loop: malformed spatialvla episode {key} {condition}")
            continue
        grouped[key][condition] = row
    if len(rows) != 1984 or set(grouped) != expected_pairs or any(
            set(group) != condition_set for group in grouped.values()):
        issue(issues, "closed loop: spatialvla lacks 496 resets across all four conditions")
    for key, group in grouped.items():
        if len(group) == 4 and len({row["initial_observation_sha256"] for row in group.values()}) != 1:
            issue(issues, f"closed loop: spatialvla reset differs across conditions at {key}")
            break

    summary_path = Path(record.get("paired_summary_path", ""))
    summary = read_json(summary_path) if summary_path.is_file() else None
    if (not summary or sha256(summary_path) != record.get("paired_summary_sha256")
            or summary.get("model_id") != "spatialvla"
            or summary.get("evidence_tier") != "full_paired_closed_loop"
            or summary.get("task_count") != 12
            or summary.get("initializations_per_condition") != 496
            or summary.get("paired_episode_rows") != 1984
            or summary.get("episodes_sha256") != sha256(episodes_path)
            or summary.get("task_manifest_sha256") != sha256(task_manifest_path)):
        issue(issues, "closed loop: spatialvla paired summary cardinality/provenance differs")
        return
    manifest_sources = summary.get("condition_manifests") or {}
    official_sources = summary.get("condition_eval_info") or {}
    if set(manifest_sources) != condition_set or set(official_sources) != condition_set:
        issue(issues, "closed loop: spatialvla lacks four condition provenance records")
        return
    client_path = Path(__file__).with_name("run_spatialvla_paired_client.py")
    server_path = Path(__file__).with_name("run_spatialvla_paired_server.py")
    client_hash, server_hash = sha256(client_path), sha256(server_path)
    reconstructed = defaultdict(dict)
    for condition in conditions:
        batch = 1 if condition.endswith("singleton") else 4
        item = manifest_sources[condition]
        manifest_path = Path(item.get("path", ""))
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if (not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256")
                or manifest.get("model") != "spatialvla"
                or manifest.get("condition") != condition
                or manifest.get("batch_size") != batch
                or manifest.get("invariant_ops") != condition.startswith("patched")
                or manifest.get("seed") != 20261005
                or manifest.get("task_manifest_sha256") != sha256(task_manifest_path)
                or manifest.get("task_count") != 12
                or manifest.get("planned_initializations") != 496
                or manifest.get("deterministic_policy") is not True
                or manifest.get("explicit_physical_configuration") is not True
                or manifest.get("client_source_sha256") != client_hash
                or manifest.get("server_source_sha256") != server_hash):
            issue(issues, f"closed loop: spatialvla condition manifest differs for {condition}")
            continue
        metadata_path = Path(manifest.get("server_metadata_path", ""))
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        if (not metadata_path.is_file()
                or sha256(metadata_path) != manifest.get("server_metadata_sha256")
                or metadata.get("policy_type") != "spatialvla"
                or metadata.get("checkpoint") != str(SPATIALVLA_CHECKPOINT.resolve())
                or metadata.get("task_suite") != task_manifest.get("task_suite")
                or metadata.get("paired_condition") != condition
                or metadata.get("maximum_request_batch") != batch
                or metadata.get("batch_invariant_ops") != condition.startswith("patched")
                or metadata.get("genuine_batched_policy_call") is not True
                or metadata.get("deterministic_policy") is not True
                or metadata.get("artifact_files") != expected_artifacts):
            issue(issues, f"closed loop: spatialvla server handshake differs for {condition}")
        expected_groups = {
            (task["task_id"], tuple(range(start, min(start + batch, task["initializations"]))))
            for task in tasks for start in range(0, task["initializations"], batch)
        }
        observed_groups = set()
        for rollout_path in rollout_json_paths(manifest_path.parent / "rollouts"):
            payload = read_json(rollout_path) or {}
            group_key = (payload.get("task_id"), tuple(payload.get("explicit_initialization_ids", [])))
            raw_episodes = payload.get("episodes")
            if (group_key not in expected_groups or group_key in observed_groups
                    or payload.get("seeds") != [20261005 + value for value in group_key[1]]
                    or not isinstance(raw_episodes, list) or len(raw_episodes) != len(group_key[1])):
                issue(issues, f"closed loop: spatialvla malformed physical group {rollout_path}")
                continue
            observed_groups.add(group_key)
            for raw_row in raw_episodes:
                key = (group_key[0], raw_row.get("initialization_id"))
                if key in reconstructed[condition]:
                    issue(issues, f"closed loop: spatialvla duplicate raw episode {condition} {key}")
                    continue
                reconstructed[condition][key] = raw_row
                combined = grouped.get(key, {}).get(condition)
                expected_combined = dict(raw_row)
                expected_combined.update({
                    "condition": condition,
                    "source_rollout_path": str(rollout_path.resolve()),
                    "explicit_initialization_id": key[1],
                })
                if combined != expected_combined:
                    issue(issues, f"closed loop: spatialvla reduced/raw episode differs at {condition} {key}")
                    break
        if observed_groups != expected_groups or set(reconstructed[condition]) != expected_pairs:
            issue(issues, f"closed loop: spatialvla raw physical schedule differs for {condition}")
        info_item = official_sources[condition]
        info_path = Path(info_item.get("path", ""))
        info = read_json(info_path) if info_path.is_file() else {}
        successes = sum(bool(grouped[key][condition]["success"]) for key in expected_pairs if condition in grouped.get(key, {}))
        if (not info_path.is_file() or sha256(info_path) != info_item.get("sha256")
                or info.get("condition") != condition
                or info.get("task_suite") != task_manifest.get("task_suite")
                or info.get("overall", {}).get("n_episodes") != 496
                or info.get("overall", {}).get("n_success") != successes):
            issue(issues, f"closed loop: spatialvla official metrics differ for {condition}")

    expected_success = {
        condition: sum(bool(grouped[key][condition]["success"]) for key in expected_pairs if condition in grouped.get(key, {}))
        for condition in conditions
    }
    pairs = list(itertools.combinations(conditions, 2))
    expected_disagreements, expected_divergence = {}, {}
    for pair_index, (left, right) in enumerate(pairs):
        name = f"{left}|{right}"
        ordered = sorted(expected_pairs)
        if any(set(grouped.get(key, {})) != condition_set for key in ordered):
            continue
        differences = [int(grouped[key][right]["success"]) - int(grouped[key][left]["success"]) for key in ordered]
        expected_disagreements[name] = sum(value != 0 for value in differences)
        expected_divergence[name] = sum(
            grouped[key][left]["action_sequence"] != grouped[key][right]["action_sequence"]
            for key in ordered
        )
        reported = (summary.get("paired_confidence_intervals") or {}).get(name, {})
        generator = random.Random(20261005 + pair_index); count = len(differences)
        means = sorted(sum(differences[generator.randrange(count)] for _ in range(count)) / count for _ in range(10_000))
        def percentile(fraction: float) -> float:
            position = fraction * (len(means) - 1); lower = int(position)
            upper = min(lower + 1, len(means) - 1)
            return means[lower] + (means[upper] - means[lower]) * (position - lower)
        expected_ci = [percentile(0.025), percentile(0.975)]
        if (reported.get("method") != "paired_percentile_bootstrap"
                or reported.get("resamples") != 10_000
                or reported.get("seed") != 20261005 + pair_index
                or abs(reported.get("difference", float("inf")) - statistics.fmean(differences)) > 1e-12
                or reported.get("ci95") != expected_ci):
            issue(issues, f"closed loop: spatialvla paired CI differs for {name}")
    if (summary.get("success_counts") != expected_success
            or summary.get("disagreement_counts") != expected_disagreements
            or summary.get("action_divergence_counts") != expected_divergence):
        issue(issues, "closed loop: spatialvla paired outcomes differ from raw episodes")


def cogact_evaluator_evidence_audit(
    record: dict, episodes_path: Path, issues: list[str],
) -> None:
    """Independently reconstruct the stochastic 12-task CogACT campaign."""
    conditions = (
        "native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic",
    )
    condition_set = set(conditions)
    task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
    if (not task_manifest_path.is_file()
            or sha256(task_manifest_path) != COGACT_TASK_MANIFEST_SHA256
            or record.get("evaluator_task_manifest_sha256") != COGACT_TASK_MANIFEST_SHA256):
        issue(issues, "closed loop: cogact_small lacks the pinned evaluator task manifest")
        return
    task_manifest = read_json(task_manifest_path) or {}
    tasks = task_manifest.get("tasks") or []
    if (task_manifest.get("schema_version") != 1
            or task_manifest.get("model_id") != "cogact_small"
            or task_manifest.get("task_suite") != "simplerenv_cogact_semantic_12_variant"
            or task_manifest.get("cogact_repository_revision") != "b174a1b86deedfab4d198d935207e7bb0527994e"
            or task_manifest.get("simplerenv_repository_revision") != "ccfe3809766839a2fcfb7a3d3c9abff585189188"
            or task_manifest.get("base_seed") != 20261005
            or task_manifest.get("conditions") != list(conditions)
            or task_manifest.get("task_count") != 12
            or task_manifest.get("initializations_per_condition") != 496
            or task_manifest.get("logical_episodes_all_conditions") != 1984
            or len(tasks) != 12
            or [task.get("task_id") for task in tasks] != list(range(12))
            or [task.get("initializations") for task in tasks] != [24] * 4 + [50] * 8
            or [task.get("policy_setup") for task in tasks]
            != ["widowx_bridge"] * 4 + ["google_robot"] * 8):
        issue(issues, "closed loop: cogact_small pinned 12-task schedule differs")
        return
    for source in task_manifest.get("source_files", []):
        source_path = Path(source.get("path", ""))
        if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
            issue(issues, f"closed loop: cogact_small evaluator source changed: {source_path}")

    checkpoint = COGACT_CHECKPOINT.resolve()
    expected_files = [{"path": str(checkpoint), "sha256": COGACT_CHECKPOINT_SHA256}]
    if (record.get("exact_checkpoint_files") != expected_files
            or record.get("checkpoint_weight_path") != str(checkpoint)
            or record.get("checkpoint_weight_sha256") != COGACT_CHECKPOINT_SHA256):
        issue(issues, "closed loop: cogact_small exact checkpoint differs")
    elif checkpoint.is_file():
        if (checkpoint.stat().st_size != COGACT_CHECKPOINT_BYTES
                or sha256(checkpoint) != COGACT_CHECKPOINT_SHA256):
            issue(issues, "closed loop: cogact_small exact checkpoint differs")
    else:
        # The 30 GB /dev/shm file is a reproducible cache, not experiment
        # evidence.  Permit its post-campaign eviction only when a strict
        # receipt pins the artifact and every retained execution/output source
        # that established the digest during the completed campaign.
        eviction_path = episodes_path.parent.parent / "checkpoint_cache_eviction.json"
        eviction = read_json(eviction_path) if eviction_path.is_file() else {}
        evidence = eviction.get("evidence") if isinstance(eviction, dict) else None
        expected_evidence = {
            episodes_path.parent / condition / filename
            for condition in conditions
            for filename in ("condition_manifest.json", "server_metadata.json")
        } | {
            episodes_path,
            episodes_path.parent / "paired_summary.json",
        }
        observed_evidence = {
            Path(item.get("path", "")): item.get("sha256")
            for item in evidence or [] if isinstance(item, dict)
        }
        if (eviction.get("schema_version") != 1
                or eviction.get("model_id") != "cogact_small"
                or eviction.get("cache_artifact_removed") is not True
                or eviction.get("experiment_evidence_removed") is not False
                or eviction.get("checkpoint_path") != str(checkpoint)
                or eviction.get("checkpoint_size_bytes") != COGACT_CHECKPOINT_BYTES
                or eviction.get("checkpoint_sha256") != COGACT_CHECKPOINT_SHA256
                or eviction.get("checkpoint_repository_revision")
                    != record.get("checkpoint_repository_revision")
                or not isinstance(eviction.get("recorded_at_utc"), str)
                or not eviction.get("recorded_at_utc")
                or set(observed_evidence) != expected_evidence
                or any(not source.is_file() or sha256(source) != expected
                       for source, expected in observed_evidence.items())):
            issue(issues, "closed loop: cogact_small evicted checkpoint cache lacks exact retained provenance")
    if record.get("paired_start_seed") != 20261005:
        issue(issues, "closed loop: cogact_small reset seed schedule differs")

    expected_pairs = {
        (task["task_id"], initialization)
        for task in tasks for initialization in range(task["initializations"])
    }
    max_steps = (120, 120, 120, 120, 113, 113, 113, 113, 113, 113, 80, 80)

    def finite_vector(value: object, length: int | None = None) -> bool:
        return (isinstance(value, list) and bool(value)
                and (length is None or len(value) == length)
                and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                        and np.isfinite(item) for item in value))

    def finite_matrix(value: object, rows: int, columns: int) -> bool:
        return (isinstance(value, list) and len(value) == rows
                and all(finite_vector(row, columns) for row in value))

    def expected_noise(task: int, initialization: int, step: int) -> str:
        sequence = np.random.SeedSequence(
            [20261005, task, initialization, step, 0xC06AC7]
        )
        noise = np.random.Generator(np.random.PCG64(sequence)).standard_normal(
            (16, 7), dtype=np.float32,
        )
        return hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()

    rows = list(stream_jsonl(episodes_path))
    grouped: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        task = row.get("task"); initialization = row.get("initialization_id")
        key = (task, initialization); condition = row.get("condition")
        action = row.get("action_sequence"); raw = row.get("raw_action_sequence")
        chunks = row.get("raw_action_chunks"); trajectory = row.get("trajectory")
        noise_hashes = row.get("noise_sha256"); rewards = row.get("rewards")
        done = row.get("done"); truncated = row.get("truncated")
        batch_sizes = row.get("policy_batch_sizes")
        length = len(action) if isinstance(action, list) else -1
        expected_hashes = (
            [expected_noise(task, initialization, step) for step in range(length)]
            if key in expected_pairs and length > 0 else []
        )
        pose_key = COGACT_POSE_KEYS[task] if isinstance(task, int) and 0 <= task < 12 else None
        valid_trajectory = (
            isinstance(trajectory, list) and len(trajectory) == length + 1
            and all(isinstance(state, dict) and set(state) == {"qpos", "qvel", pose_key}
                    and finite_vector(state["qpos"]) and finite_vector(state["qvel"])
                    and len(state["qpos"]) == len(state["qvel"])
                    and finite_vector(state[pose_key]) for state in trajectory)
        )
        if (key not in expected_pairs or condition not in condition_set
                or condition in grouped[key]
                or row.get("task_name") != tasks[task]["name"]
                or row.get("task_group") != "simplerenv_cogact_semantic_12_variant"
                or row.get("policy_setup") != tasks[task]["policy_setup"]
                or not isinstance(row.get("task_description"), str)
                or not row["task_description"]
                or row.get("reset_seed") != 20261005 + initialization
                or row.get("explicit_initialization_id") != initialization
                or not is_sha256_hex(row.get("initial_observation_sha256"))
                or not isinstance(row.get("success"), bool)
                or length <= 0 or length > max_steps[task]
                or not all(finite_vector(item, 7) for item in action)
                or not isinstance(raw, list) or len(raw) != length
                or not all(finite_vector(item, 7) for item in raw)
                or not isinstance(chunks, list) or len(chunks) != length
                or not all(finite_matrix(item, 16, 7) for item in chunks)
                or noise_hashes != expected_hashes or len(set(noise_hashes)) != length
                or not valid_trajectory
                or not isinstance(rewards, list) or len(rewards) != length
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and np.isfinite(value) for value in rewards)
                or not isinstance(done, list) or len(done) != length
                or not all(isinstance(value, bool) for value in done)
                or not isinstance(truncated, list) or len(truncated) != length
                or not all(isinstance(value, bool) for value in truncated)
                or not truncated[-1] or row["success"] != any(done)
                or not isinstance(batch_sizes, list) or len(batch_sizes) != length
                or not all(isinstance(value, int) and 1 <= value <= 4 for value in batch_sizes)):
            issue(issues, f"closed loop: malformed cogact_small episode {key} {condition}")
            continue
        grouped[key][condition] = row
    if (len(rows) != 1984 or set(grouped) != expected_pairs
            or any(set(group) != condition_set for group in grouped.values())):
        issue(issues, "closed loop: cogact_small lacks 496 resets across all four conditions")
    for key, group in grouped.items():
        if len(group) == 4 and (
            len({row["initial_observation_sha256"] for row in group.values()}) != 1
            or len({row["reset_seed"] for row in group.values()}) != 1
            or len({json.dumps(row["noise_sha256"]) for row in group.values()}) != 1
        ):
            issue(issues, f"closed loop: cogact_small reset/noise differs at {key}")
            break

    summary_path = Path(record.get("paired_summary_path", ""))
    summary = read_json(summary_path) if summary_path.is_file() else None
    if (not summary or sha256(summary_path) != record.get("paired_summary_sha256")
            or summary.get("model_id") != "cogact_small"
            or summary.get("evidence_tier") != "full_paired_closed_loop"
            or summary.get("conditions") != list(conditions)
            or summary.get("task_count") != 12
            or summary.get("initializations_per_condition") != 496
            or summary.get("paired_episode_rows") != 1984
            or summary.get("episodes_sha256") != sha256(episodes_path)
            or summary.get("task_manifest_sha256") != COGACT_TASK_MANIFEST_SHA256):
        issue(issues, "closed loop: cogact_small paired summary cardinality/provenance differs")
        return
    pairing_path = Path(record.get("reset_pairing_audit_path", ""))
    pairing = read_json(pairing_path) if pairing_path.is_file() else None
    if (not pairing or sha256(pairing_path) != record.get("reset_pairing_audit_sha256")
            or pairing.get("schema_version") != 1
            or pairing.get("model_id") != "cogact_small"
            or pairing.get("evidence_tier") != "independent_raw_pairing_audit"
            or pairing.get("task_manifest_sha256") != COGACT_TASK_MANIFEST_SHA256
            or pairing.get("expected_resets_per_condition") != 496
            or pairing.get("initial_observation_or_seed_mismatch_count") != 0
            or pairing.get("request_local_noise_mismatch_count") != 0
            or pairing.get("fully_paired") is not True
            or pairing.get("issues") != []):
        issue(issues, "closed loop: cogact_small independent reset/noise pairing audit differs")
    manifest_sources = summary.get("condition_manifests") or {}
    official_sources = summary.get("condition_eval_info") or {}
    if set(manifest_sources) != condition_set or set(official_sources) != condition_set:
        issue(issues, "closed loop: cogact_small lacks four condition provenance records")
        return

    reconstructed: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for condition in conditions:
        batch = 1 if condition.endswith("singleton") else 4
        repaired = condition.startswith("patched")
        item = manifest_sources[condition]
        manifest_path = Path(item.get("path", ""))
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if (not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256")
                or manifest.get("model") != "cogact_small"
                or manifest.get("condition") != condition
                or manifest.get("batch_size") != batch
                or manifest.get("invariant_ops") != repaired
                or manifest.get("seed") != 20261005
                or manifest.get("task_suite") != task_manifest.get("task_suite")
                or manifest.get("task_manifest_sha256") != COGACT_TASK_MANIFEST_SHA256
                or manifest.get("task_count") != 12
                or manifest.get("planned_initializations") != 496
                or manifest.get("request_local_noise") is not True
                or manifest.get("noise_shape") != [16, 7]
                or manifest.get("noise_generator")
                != "numpy.PCG64.SeedSequence([seed,task,initialization,step,0xC06AC7])"
                or manifest.get("explicit_physical_configuration") is not True
                or manifest.get("checkpoint_weight_path") != str(checkpoint)
                or manifest.get("checkpoint_weight_sha256") != COGACT_CHECKPOINT_SHA256
                or manifest.get("client_source_sha256") != COGACT_CLIENT_SHA256
                or manifest.get("server_source_sha256") != COGACT_SERVER_SHA256):
            issue(issues, f"closed loop: cogact_small condition manifest differs for {condition}")
            continue
        metadata_path = Path(manifest.get("server_metadata_path", ""))
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        expected_metadata = {
            "protocol_version": "2.1", "policy_type": "cogact_small",
            "checkpoint": "/workspace/assets/cogact_small",
            "checkpoint_weight_sha256": COGACT_CHECKPOINT_SHA256,
            "task_suite": "simplerenv_cogact_semantic_12_variant",
            "paired_condition": condition, "genuine_batched_policy_call": True,
            "maximum_request_batch": batch, "batch_invariant_ops": repaired,
            "request_local_noise": True, "noise_shape": [16, 7],
            "cfg_scale": 1.5, "ddim_steps": 10,
            "unnorm_keys": {"google_robot": "fractal20220817_data", "widowx_bridge": "bridge_orig"},
            "action_chunk_size": 16, "action_dim": 7, "adaptive_ensemble_alpha": 0.1,
            "llama_construction_dtype": "torch.bfloat16",
            "llama_random_weight_initialization": "skipped_before_strict_checkpoint_load",
            "final_vlm_dtype": "torch.bfloat16", "final_action_head_dtype": "torch.float32",
        }
        if (not metadata_path.is_file()
                or sha256(metadata_path) != manifest.get("server_metadata_sha256")
                or metadata != expected_metadata):
            issue(issues, f"closed loop: cogact_small server handshake differs for {condition}")
        expected_groups = {
            (task["task_id"], tuple(range(start, min(start + batch, task["initializations"]))))
            for task in tasks for start in range(0, task["initializations"], batch)
        }
        observed_groups = set()
        for rollout_path in rollout_json_paths(manifest_path.parent / "rollouts"):
            payload = read_json(rollout_path) or {}
            group_key = (payload.get("task_id"), tuple(payload.get("explicit_initialization_ids", [])))
            raw_episodes = payload.get("episodes")
            if (group_key not in expected_groups or group_key in observed_groups
                    or payload.get("seeds") != [20261005 + value for value in group_key[1]]
                    or not isinstance(raw_episodes, list)
                    or len(raw_episodes) != len(group_key[1])):
                issue(issues, f"closed loop: cogact_small malformed physical group {rollout_path}")
                continue
            observed_groups.add(group_key)
            for raw_row in raw_episodes:
                key = (group_key[0], raw_row.get("initialization_id"))
                if key in reconstructed[condition]:
                    issue(issues, f"closed loop: cogact_small duplicate raw episode {condition} {key}")
                    continue
                reconstructed[condition][key] = raw_row
                combined = grouped.get(key, {}).get(condition)
                expected_combined = dict(raw_row)
                expected_combined.update({
                    "condition": condition,
                    "source_rollout_path": str(rollout_path.resolve()),
                    "explicit_initialization_id": key[1],
                })
                if combined != expected_combined:
                    issue(issues, f"closed loop: cogact_small reduced/raw episode differs at {condition} {key}")
                    break
        if observed_groups != expected_groups or set(reconstructed[condition]) != expected_pairs:
            issue(issues, f"closed loop: cogact_small raw physical schedule differs for {condition}")
        info_item = official_sources[condition]
        info_path = Path(info_item.get("path", ""))
        info = read_json(info_path) if info_path.is_file() else {}
        successes = sum(bool(grouped[key][condition]["success"])
                        for key in expected_pairs if condition in grouped.get(key, {}))
        if (not info_path.is_file() or sha256(info_path) != info_item.get("sha256")
                or info.get("condition") != condition
                or info.get("task_suite") != task_manifest.get("task_suite")
                or info.get("overall", {}).get("n_episodes") != 496
                or info.get("overall", {}).get("n_success") != successes):
            issue(issues, f"closed loop: cogact_small official metrics differ for {condition}")

    if all(set(grouped.get(key, {})) == condition_set for key in expected_pairs):
        ordered = sorted(expected_pairs)
        expected_success = {
            condition: sum(bool(grouped[key][condition]["success"]) for key in ordered)
            for condition in conditions
        }
        expected_disagreement = {}; expected_action = {}; expected_trajectory = {}
        expected_details = {}
        for pair_index, (left, right) in enumerate(itertools.combinations(conditions, 2)):
            name = f"{left}|{right}"
            differences = [int(grouped[key][right]["success"])
                           - int(grouped[key][left]["success"]) for key in ordered]
            action_first = [] ; trajectory_first = []
            for key in ordered:
                left_action = grouped[key][left]["action_sequence"]
                right_action = grouped[key][right]["action_sequence"]
                first = next((index for index, values in enumerate(zip(left_action, right_action))
                              if values[0] != values[1]), None)
                if first is None and len(left_action) != len(right_action):
                    first = min(len(left_action), len(right_action))
                action_first.append(first)
                left_trajectory = grouped[key][left]["trajectory"]
                right_trajectory = grouped[key][right]["trajectory"]
                first = next((index for index, values in enumerate(zip(left_trajectory, right_trajectory))
                              if values[0] != values[1]), None)
                if first is None and len(left_trajectory) != len(right_trajectory):
                    first = min(len(left_trajectory), len(right_trajectory))
                trajectory_first.append(first)
            expected_disagreement[name] = sum(value != 0 for value in differences)
            expected_action[name] = sum(value is not None for value in action_first)
            expected_trajectory[name] = sum(value is not None for value in trajectory_first)
            expected_details[name] = {
                "first_action_divergence_step_by_reset": action_first,
                "first_trajectory_divergence_state_by_reset": trajectory_first,
            }
            reported = (summary.get("paired_confidence_intervals") or {}).get(name, {})
            generator = random.Random(20261005 + pair_index); count = len(differences)
            means = sorted(sum(differences[generator.randrange(count)] for _ in range(count)) / count
                           for _ in range(10_000))
            position_low = .025 * (len(means) - 1); low = int(position_low)
            position_high = .975 * (len(means) - 1); high = int(position_high)
            expected_ci = [
                means[low] + (means[min(low + 1, len(means) - 1)] - means[low]) * (position_low - low),
                means[high] + (means[min(high + 1, len(means) - 1)] - means[high]) * (position_high - high),
            ]
            if (reported.get("method") != "paired_percentile_bootstrap"
                    or reported.get("resamples") != 10_000
                    or reported.get("seed") != 20261005 + pair_index
                    or abs(reported.get("difference", float("inf"))
                           - statistics.fmean(differences)) > 1e-12
                    or reported.get("ci95") != expected_ci):
                issue(issues, f"closed loop: cogact_small paired CI differs for {name}")
        if (summary.get("success_counts") != expected_success
                or summary.get("disagreement_counts") != expected_disagreement
                or summary.get("action_divergence_counts") != expected_action
                or summary.get("trajectory_divergence_counts") != expected_trajectory
                or summary.get("divergence_details") != expected_details):
            issue(issues, "closed loop: cogact_small paired outcomes differ from raw episodes")


OPENHELIX_PAIRED_CONDITIONS = (
    "native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic",
)


def openhelix_array_bundle_sha256(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def openhelix_request_seed(sequence: int, subtask: int, policy_call: int) -> int:
    payload = f"20261015:{sequence}:{subtask}:{policy_call}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (2**63 - 1)


def openhelix_target_position(batch_size: int, policy_call: int) -> int:
    label = ("first", "middle", "last")[policy_call % 3]
    if label == "first" or batch_size == 1:
        return 0
    if label == "middle":
        return batch_size // 2
    return batch_size - 1


def openhelix_load_trajectory(path: Path, expected_sha256: str, issues: list[str]) -> dict[str, np.ndarray] | None:
    if not path.is_file() or sha256(path) != expected_sha256:
        issue(issues, f"closed loop: OpenHelix trajectory archive changed: {path}")
        return None
    try:
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
    except (OSError, TypeError, ValueError) as error:
        issue(issues, f"closed loop: OpenHelix trajectory archive cannot be decoded: {path}: {error}")
        return None
    expected = {
        "actions", "robot_states", "scene_states", "rewards", "dones",
        "subtask_indices", "policy_call_indices", "action_chunk_indices",
        "batch_sizes", "target_positions",
    }
    lengths = {value.shape[0] for value in arrays.values()}
    if (set(arrays) != expected or len(lengths) != 1
            or arrays.get("actions", np.empty((0,))).shape[1:] != (7,)
            or arrays.get("robot_states", np.empty((0,))).shape[1:] != (15,)
            or arrays.get("scene_states", np.empty((0,))).shape[1:] != (24,)
            or arrays.get("actions", np.empty(0)).dtype != np.float32
            or arrays.get("robot_states", np.empty(0)).dtype != np.float32
            or arrays.get("scene_states", np.empty(0)).dtype != np.float32
            or any(not np.isfinite(arrays[name]).all() for name in (
                "actions", "robot_states", "scene_states", "rewards"
            ))):
        issue(issues, f"closed loop: OpenHelix trajectory shapes/dtypes/values differ: {path}")
        return None
    return arrays


def openhelix_first_divergence(left: np.ndarray, right: np.ndarray) -> int | None:
    common = min(len(left), len(right))
    if common:
        unequal = np.flatnonzero(np.any(left[:common] != right[:common], axis=1))
        if unequal.size:
            return int(unequal[0])
    return common if len(left) != len(right) else None


def openhelix_paired_bootstrap(differences: list[int], seed: int) -> dict:
    generator = random.Random(seed)
    count = len(differences)
    means = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(10_000)
    )

    def percentile(fraction: float) -> float:
        position = fraction * (len(means) - 1)
        lower = int(position)
        upper = min(lower + 1, len(means) - 1)
        return means[lower] + (means[upper] - means[lower]) * (position - lower)

    return {
        "method": "paired_percentile_bootstrap", "resamples": 10_000,
        "seed": seed, "difference": statistics.fmean(differences),
        "ci95": [percentile(0.025), percentile(0.975)],
    }


def openhelix_evaluator_evidence_audit(
    record: dict, episodes_file: Path, issues: list[str]
) -> None:
    """Independently validate the official 4 x 1,000 CALVIN sequence campaign."""
    if sha256(episodes_file) != record.get("episodes_sha256"):
        issue(issues, "closed loop: OpenHelix combined episode stream hash differs")
        return
    policy = Path(record.get("checkpoint_weight_path", ""))
    source = Path(record.get("official_evaluator_source_path", ""))
    wrapper = Path(record.get("paired_wrapper_path", ""))
    if (not policy.is_file() or policy.stat().st_size != 445_004_793
            or sha256(policy) != OPENHELIX_CHECKPOINT_FILES[-1][2]
            or record.get("checkpoint_weight_sha256") != OPENHELIX_CHECKPOINT_FILES[-1][2]):
        issue(issues, "closed loop: OpenHelix exact policy checkpoint differs")
    if (not source.is_file() or sha256(source) != record.get("official_evaluator_source_sha256")
            or source != Path(record.get("source_evidence_path", ""))
            or record.get("source_evidence_sha256") != record.get("official_evaluator_source_sha256")):
        issue(issues, "closed loop: OpenHelix official evaluator source differs")
    if not wrapper.is_file() or sha256(wrapper) != record.get("paired_wrapper_sha256"):
        issue(issues, "closed loop: OpenHelix paired wrapper differs")

    rows = list(stream_jsonl(episodes_file))
    expected_keys = {
        (sequence, condition)
        for sequence in range(1000) for condition in OPENHELIX_PAIRED_CONDITIONS
    }
    by_key = {}
    trajectories = {}
    for row in rows:
        sequence = row.get("sequence_index")
        condition = row.get("condition")
        key = (sequence, condition)
        if key not in expected_keys or key in by_key:
            issue(issues, "closed loop: OpenHelix has an unexpected or duplicate paired record")
            break
        by_key[key] = row
        metadata_path = Path(row.get("source_metadata_path", ""))
        trajectory_path = Path(row.get("trajectory_npz_path", ""))
        if (not metadata_path.is_file()
                or sha256(metadata_path) != row.get("source_metadata_sha256")):
            issue(issues, f"closed loop: OpenHelix raw rollout metadata changed: {key}")
            break
        try:
            with gzip.open(metadata_path, "rt") as stream:
                raw = json.load(stream)
        except (OSError, ValueError) as error:
            issue(issues, f"closed loop: OpenHelix raw rollout metadata cannot be decoded: {key}: {error}")
            break
        expected_combined = dict(raw)
        expected_combined.update({
            "source_metadata_path": str(metadata_path.resolve()),
            "source_metadata_sha256": row.get("source_metadata_sha256"),
        })
        if row != expected_combined:
            issue(issues, f"closed loop: OpenHelix combined record differs from raw rollout: {key}")
            break
        if (row.get("initialization_id") != sequence
                or row.get("explicit_init_state_id") != sequence
                or not is_sha256_hex(row.get("initial_condition_sha256"))
                or not is_sha256_hex(row.get("initial_observation_sha256"))
                or not is_sha256_hex(row.get("first_noise_sha256"))
                or row.get("sequence_length") != 5
                or not isinstance(row.get("task_sequence"), list)
                or len(row["task_sequence"]) != 5
                or not isinstance(row.get("tasks_completed"), int)
                or not 0 <= row["tasks_completed"] <= 5
                or row.get("success") is not (row["tasks_completed"] == 5)
                or row.get("policy_calls") != len(row.get("request_schedule") or [])
                or row.get("request_noise_sha256")
                    != [item.get("noise_sha256") for item in row.get("request_schedule") or []]
                or row.get("request_schedule_sha256")
                    != canonical_sha256(row.get("request_schedule") or [])):
            issue(issues, f"closed loop: OpenHelix outcome/reset/request evidence differs: {key}")
            break
        schedule = row["request_schedule"]
        if not schedule or row.get("first_noise_sha256") != schedule[0].get("noise_sha256"):
            issue(issues, f"closed loop: OpenHelix lacks first request noise: {key}")
            break
        dynamic = condition.endswith("dynamic")
        for request in schedule:
            subtask = request.get("subtask_index")
            policy_call = request.get("policy_call")
            if (not isinstance(subtask, int) or not 0 <= subtask < 5
                    or not isinstance(policy_call, int) or not 0 <= policy_call < 60):
                issue(issues, f"closed loop: OpenHelix request identity differs: {key}")
                break
            batch_size = (1, 2, 4, 8)[policy_call % 4] if dynamic else 1
            if (request.get("noise_seed") != openhelix_request_seed(sequence, subtask, policy_call)
                    or request.get("batch_size") != batch_size
                    or request.get("target_position")
                        != (openhelix_target_position(batch_size, policy_call) if dynamic else 0)
                    or not is_sha256_hex(request.get("noise_sha256"))):
                issue(issues, f"closed loop: OpenHelix batching/noise schedule differs: {key}")
                break
        if issues and issues[-1].endswith(f": {key}"):
            break
        arrays = openhelix_load_trajectory(
            trajectory_path, row.get("trajectory_npz_sha256", ""), issues
        )
        if arrays is None:
            break
        schedule_by_call = {
            (item["subtask_index"], item["policy_call"]): item
            for item in schedule
        }
        action_call_keys = list(zip(
            arrays["subtask_indices"].tolist(),
            arrays["policy_call_indices"].tolist(),
        ))
        if (len(schedule_by_call) != len(schedule)
                or any(call not in schedule_by_call for call in action_call_keys)
                or any(not 0 <= int(index) < 19 for index in arrays["action_chunk_indices"])
                or not np.array_equal(
                    arrays["batch_sizes"],
                    np.asarray([
                        schedule_by_call[call]["batch_size"] for call in action_call_keys
                    ], dtype=np.int8),
                )
                or not np.array_equal(
                    arrays["target_positions"],
                    np.asarray([
                        schedule_by_call[call]["target_position"] for call in action_call_keys
                    ], dtype=np.int8),
                )):
            issue(issues, f"closed loop: OpenHelix physical action/request assignment differs: {key}")
            break
        if (row.get("trajectory_npz_path") != str(trajectory_path.resolve())
                or row.get("environment_steps") != arrays["actions"].shape[0]
                or row.get("trajectory_arrays_sha256") != openhelix_array_bundle_sha256(arrays)):
            issue(issues, f"closed loop: OpenHelix trajectory metadata differs: {key}")
            break
        trajectories[key] = {
            "path": trajectory_path,
            "action_hash": hashlib.sha256(arrays["actions"].tobytes()).hexdigest(),
            "robot_hash": hashlib.sha256(arrays["robot_states"].tobytes()).hexdigest(),
            "scene_hash": hashlib.sha256(arrays["scene_states"].tobytes()).hexdigest(),
        }
    if set(by_key) != expected_keys or len(rows) != 4000:
        issue(issues, f"closed loop: OpenHelix has {len(by_key)}/4000 unique paired sequence records")
        return

    reset_mismatches = []
    for sequence in range(1000):
        group = [by_key[(sequence, condition)] for condition in OPENHELIX_PAIRED_CONDITIONS]
        if len({row["initial_condition_sha256"] for row in group}) != 1:
            issue(issues, f"closed loop: OpenHelix initial condition differs at sequence {sequence}")
            return
        if len({row["first_noise_sha256"] for row in group}) != 1:
            issue(issues, f"closed loop: OpenHelix first request noise differs at sequence {sequence}")
            return
        if len({row["initial_observation_sha256"] for row in group}) != 1:
            reset_mismatches.append(sequence)

    summary_path = Path(record.get("paired_summary_path", ""))
    if (not summary_path.is_file() or sha256(summary_path) != record.get("paired_summary_sha256")):
        issue(issues, "closed loop: OpenHelix paired summary is missing or changed")
        return
    summary = read_json(summary_path) or {}
    manifests = summary.get("condition_manifests") or {}
    if set(manifests) != set(OPENHELIX_PAIRED_CONDITIONS):
        issue(issues, "closed loop: OpenHelix lacks four condition manifests")
    for condition in OPENHELIX_PAIRED_CONDITIONS:
        item = manifests.get(condition) or {}
        manifest_path = Path(item.get("path", ""))
        if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
            issue(issues, f"closed loop: OpenHelix condition manifest differs: {condition}")
            continue
        manifest = read_json(manifest_path) or {}
        repaired = condition.startswith("patched")
        dynamic = condition.endswith("dynamic")
        if (manifest.get("model") != "openhelix"
                or manifest.get("condition") != condition
                or manifest.get("invariant_ops") is not repaired
                or manifest.get("execution_mode")
                    != ("dynamic_batch" if dynamic else "serial_singleton")
                or manifest.get("sequence_start") != 0
                or manifest.get("sequence_count") != 1000
                or manifest.get("official_sequence_count") != 1000
                or manifest.get("policy_calls_per_subtask") != 60
                or manifest.get("execute_len") != 20
                or manifest.get("request_local_noise") is not True
                or manifest.get("paired_noise_seed") != 20261015
                or manifest.get("policy_sha256") != OPENHELIX_CHECKPOINT_FILES[-1][2]
                or manifest.get("official_evaluator_sha256")
                    != record.get("official_evaluator_source_sha256")
                or manifest.get("paired_wrapper_sha256") != record.get("paired_wrapper_sha256")):
            issue(issues, f"closed loop: OpenHelix condition provenance differs: {condition}")

    condition_metrics = {}
    for condition in OPENHELIX_PAIRED_CONDITIONS:
        values = [by_key[(sequence, condition)]["tasks_completed"] for sequence in range(1000)]
        condition_metrics[condition] = {
            "sequences": 1000,
            "task_threshold_success_counts": {
                str(threshold): sum(value >= threshold for value in values)
                for threshold in range(1, 6)
            },
            "task_threshold_success_rates": {
                str(threshold): sum(value >= threshold for value in values) / 1000
                for threshold in range(1, 6)
            },
            "average_sequence_length": statistics.fmean(values),
        }
    intervals = {}
    disagreements = {}
    action_divergence = {}
    trajectory_divergence = {}
    details = []
    for pair_index, (left, right) in enumerate(
        itertools.combinations(OPENHELIX_PAIRED_CONDITIONS, 2)
    ):
        pair = f"{left}|{right}"
        intervals[pair] = {}
        disagreements[pair] = {}
        action_divergence[pair] = 0
        trajectory_divergence[pair] = 0
        for threshold in range(1, 6):
            differences = [
                int(by_key[(sequence, right)]["tasks_completed"] >= threshold)
                - int(by_key[(sequence, left)]["tasks_completed"] >= threshold)
                for sequence in range(1000)
            ]
            disagreements[pair][str(threshold)] = sum(value != 0 for value in differences)
            intervals[pair][str(threshold)] = openhelix_paired_bootstrap(
                differences, 20261015 + pair_index * 10 + threshold
            )
        for sequence in range(1000):
            left_info = trajectories[(sequence, left)]
            right_info = trajectories[(sequence, right)]
            action_differs = left_info["action_hash"] != right_info["action_hash"]
            state_differs = (
                left_info["robot_hash"] != right_info["robot_hash"]
                or left_info["scene_hash"] != right_info["scene_hash"]
            )
            if action_differs:
                action_divergence[pair] += 1
            if state_differs:
                trajectory_divergence[pair] += 1
            if action_differs or state_differs:
                left_arrays = openhelix_load_trajectory(left_info["path"], sha256(left_info["path"]), issues)
                right_arrays = openhelix_load_trajectory(right_info["path"], sha256(right_info["path"]), issues)
                if left_arrays is None or right_arrays is None:
                    return
                details.append({
                    "pair": pair, "sequence_index": sequence,
                    "first_action_divergence_step": openhelix_first_divergence(
                        left_arrays["actions"], right_arrays["actions"]
                    ),
                    "first_robot_state_divergence_step": openhelix_first_divergence(
                        left_arrays["robot_states"], right_arrays["robot_states"]
                    ),
                    "first_scene_state_divergence_step": openhelix_first_divergence(
                        left_arrays["scene_states"], right_arrays["scene_states"]
                    ),
                })
    if (summary.get("schema_version") != 1
            or summary.get("evidence_tier") != "full_paired_closed_loop"
            or summary.get("model_id") != "openhelix"
            or summary.get("sequences_per_condition") != 1000
            or summary.get("paired_condition_sequence_records") != 4000
            or summary.get("episodes_path") != str(episodes_file.resolve())
            or summary.get("episodes_sha256") != sha256(episodes_file)
            or summary.get("condition_metrics") != condition_metrics
            or summary.get("paired_confidence_intervals") != intervals
            or summary.get("disagreement_counts") != disagreements
            or summary.get("action_divergence_counts") != action_divergence
            or summary.get("trajectory_divergence_counts") != trajectory_divergence
            or summary.get("divergence_details") != details
            or summary.get("cross_condition_initial_observation_mismatch_count")
                != len(reset_mismatches)
            or summary.get("cross_condition_initial_observation_mismatch_sequences")
                != reset_mismatches
            or summary.get("first_request_noise_mismatch_count") != 0):
        issue(issues, "closed loop: OpenHelix paired summary differs from raw outcomes")
    if (record.get("conditions") != list(OPENHELIX_PAIRED_CONDITIONS)
            or record.get("official_sequence_count") != 1000
            or record.get("paired_condition_sequence_records") != 4000
            or record.get("condition_metrics") != condition_metrics
            or record.get("paired_confidence_intervals") != intervals
            or record.get("disagreement_counts") != disagreements
            or record.get("action_divergence_counts") != action_divergence
            or record.get("trajectory_divergence_counts") != trajectory_divergence):
        issue(issues, "closed loop: OpenHelix completion record differs from independently recomputed outcomes")


def evaluator_audit(path: Path, identifier: str, issues: list[str]) -> dict:
    start = len(issues)
    record = read_json(path)
    if not record:
        issue(issues, f"closed loop: missing checkpoint-specific evaluator attempt {path}")
        return {"complete": False}
    if record.get("model_id") != identifier:
        issue(issues, f"closed loop: evaluator record belongs to another model: {identifier}")
    status = record.get("status")
    if status == "not_applicable":
        if identifier not in NO_MATCHED_EVALUATOR:
            issue(issues, f"closed loop: {identifier} has a published checkpoint-matched evaluator")
        evidence = record.get("source_evidence_path")
        if not record.get("reason") or not evidence or not Path(evidence).is_file() or sha256(Path(evidence)) != record.get("source_evidence_sha256"):
            issue(issues, f"closed loop: N/A lacks pinned checkpoint-matched source evidence for {identifier}")
    elif status == "probe_succeeded_requires_paired_campaign":
        info_path = record.get("probe_eval_info_path")
        if record.get("exit_code") != 0 or not info_path or not Path(info_path).is_file() or sha256(Path(info_path)) != record.get("probe_eval_info_sha256"):
            issue(issues, f"closed loop: successful probe lacks immutable official metrics for {identifier}")
        else:
            info = read_json(Path(info_path)) or {}
            overall = info.get("overall", {})
            if (record.get("probe_tasks_completed") != len(info.get("per_task", []))
                    or record.get("probe_episodes_completed") != overall.get("n_episodes")
                    or record.get("probe_successes") != overall.get("n_success")):
                issue(issues, f"closed loop: probe episode counts disagree with official metrics for {identifier}")
        issue(issues, f"closed loop: {identifier} evaluator runs, but paired four-condition campaign is pending")
    elif status == "ready_pending_model_campaign":
        if identifier != "openhelix":
            issue(issues, f"closed loop: unsupported readiness-only status for {identifier}")
        evidence_keys = (
            ("paired_preflight_path", "paired_preflight_sha256"),
            ("paired_condition_manifest_path", "paired_condition_manifest_sha256"),
            ("runtime_probe_path", "runtime_probe_sha256"),
            ("evaluator_assets_path", "evaluator_assets_sha256"),
            ("checkpoint_weight_path", "checkpoint_weight_sha256"),
            ("source_evidence_path", "source_evidence_sha256"),
            ("paired_wrapper_path", "paired_wrapper_sha256"),
        )
        for path_key, hash_key in evidence_keys:
            evidence_path = Path(record.get(path_key, ""))
            if (not evidence_path.is_file()
                    or sha256(evidence_path) != record.get(hash_key)):
                issue(issues, f"closed loop: OpenHelix readiness evidence differs: {path_key}")
        preflight = read_json(Path(record.get("paired_preflight_path", ""))) or {}
        condition_manifest = read_json(
            Path(record.get("paired_condition_manifest_path", ""))
        ) or {}
        if (preflight.get("passed") is not True
                or preflight.get("sequence_generator_count") != 1000
                or preflight.get("condition_manifest_sha256")
                    != record.get("paired_condition_manifest_sha256")):
            issue(issues, "closed loop: OpenHelix paired CALVIN preflight differs")
        if (condition_manifest.get("model") != "openhelix"
                or condition_manifest.get("condition") != "native_dynamic"
                or condition_manifest.get("official_sequence_count") != 1000
                or condition_manifest.get("sequence_count") != 1000
                or condition_manifest.get("dynamic_batch_schedule") != [1, 2, 4, 8]
                or condition_manifest.get("target_position_schedule")
                    != ["first", "middle", "last"]
                or condition_manifest.get("request_local_noise") is not True
                or condition_manifest.get("policy_calls_per_subtask") != 60
                or condition_manifest.get("execute_len") != 20):
            issue(issues, "closed loop: OpenHelix paired condition contract differs")
        if (record.get("probe_episodes_completed") != 0
                or record.get("paired_episodes_completed") != 0):
            issue(issues, "closed loop: OpenHelix readiness record invents model episodes")
        paired_root = path.parent / "paired"
        raw_sequences = sum(
            1 for condition in (
                "native_singleton", "native_dynamic",
                "patched_singleton", "patched_dynamic",
            )
            for candidate in (paired_root / condition / "rollouts").glob("sequence_*.json.gz")
            if candidate.is_file()
        )
        if raw_sequences:
            issue(
                issues,
                f"closed loop: openhelix paired campaign is partial: {raw_sequences}/4000 "
                "raw official sequence records",
            )
        else:
            issue(issues, "closed loop: openhelix evaluator is runnable, but no model-backed paired episode has completed")
    elif status == "checkpoint_stream_ready_pending_model_probe":
        if identifier != "molmoact2":
            issue(issues, f"closed loop: unsupported memory-stream readiness status for {identifier}")
        evidence_keys = (
            ("checkpoint_preflight_path", "checkpoint_preflight_sha256"),
            ("adapter_path", "adapter_sha256"),
            ("replay_manifest_path", "replay_manifest_sha256"),
            ("request_noise_path", "request_noise_sha256"),
            ("paired_wrapper_path", "paired_wrapper_sha256"),
            ("paired_preflight_path", "paired_preflight_sha256"),
            ("evaluator_task_manifest_path", "evaluator_task_manifest_sha256"),
        )
        for path_key, hash_key in evidence_keys:
            evidence_path = Path(record.get(path_key, ""))
            if not evidence_path.is_file() or sha256(evidence_path) != record.get(hash_key):
                issue(issues, f"closed loop: MolmoAct2 stream readiness evidence differs: {path_key}")
        preflight = read_json(Path(record.get("checkpoint_preflight_path", ""))) or {}
        streaming = preflight.get("memory_streaming") or {}
        meta = (streaming.get("meta_model_probe") or {}).get("result") or {}
        if (preflight.get("runnable") is not True
                or preflight.get("blockers") != []
                or len(streaming.get("remote_exact_shards") or []) != 5
                or any(not item.get("range_probe") for item in streaming.get("remote_exact_shards") or [])
                or meta != {
                    "state_tensors": 1295, "index_tensors": 1295,
                    "exact_inventory": True, "meta_parameters": 1294,
                    "meta_buffers": 3,
                }):
            issue(issues, "closed loop: MolmoAct2 exact RAM-stream preflight differs")
        paired_preflight = read_json(Path(record.get("paired_preflight_path", ""))) or {}
        task_manifest = read_json(Path(record.get("evaluator_task_manifest_path", ""))) or {}
        if (paired_preflight.get("passed") is not True
                or paired_preflight.get("model_id") != "molmoact2"
                or paired_preflight.get("adapter_sha256") != record.get("adapter_sha256")
                or paired_preflight.get("paired_wrapper_sha256")
                    != record.get("paired_wrapper_sha256")
                or paired_preflight.get("official_model_repository_revision")
                    != "66b87e64efd99dfd103241418113955cf64dfa9c"
                or paired_preflight.get("official_lerobot_revision")
                    != "e595b7902714ba51f91e47523f66f89c5181b649"):
            issue(issues, "closed loop: MolmoAct2 paired evaluator preflight differs")
        tasks = task_manifest.get("tasks") or []
        if (task_manifest.get("schema_version") != 1
                or task_manifest.get("model_id") != "molmoact2"
                or task_manifest.get("task_suite") != "libero_10"
                or task_manifest.get("task_ids") != list(range(10))
                or task_manifest.get("task_descriptions") != WSA_TASK_DESCRIPTIONS
                or task_manifest.get("initializations_per_task") != 50
                or len(tasks) != 10
                or any(row.get("task_id") != index
                       or row.get("task_description") != WSA_TASK_DESCRIPTIONS[index]
                       or row.get("initializations") != 50
                       or not Path(row.get("init_states_path", "")).is_file()
                       or sha256(Path(row.get("init_states_path", "")))
                           != row.get("init_states_sha256")
                       or not Path(row.get("bddl_path", "")).is_file()
                       or sha256(Path(row.get("bddl_path", ""))) != row.get("bddl_sha256")
                       for index, row in enumerate(tasks))):
            issue(issues, "closed loop: MolmoAct2 LIBERO-10 task manifest differs")
        if (record.get("probe_episodes_completed") != 0
                or record.get("paired_episodes_completed") != 0):
            issue(issues, "closed loop: MolmoAct2 readiness record invents model episodes")
        probe_log_path = Path(record.get("latest_model_probe_log_path", ""))
        probe_log = probe_log_path.read_text(errors="replace") if probe_log_path.is_file() else ""
        if (not probe_log_path.is_file()
                or sha256(probe_log_path) != record.get("latest_model_probe_log_sha256")
                or record.get("latest_model_probe_exit_code") != 1
                or record.get("latest_model_probe_failure_class")
                    != "policy_action_tensor_reached_incorrect_empty_postprocessor_before_environment_step"
                or "Running rollout with at most 520 steps" not in probe_log
                or "EnvTransition must be a dictionary. Got Tensor" not in probe_log):
            issue(issues, "closed loop: MolmoAct2 model-backed evaluator probe evidence differs")
        issue(issues, "closed loop: MolmoAct2 exact model call succeeded, but no model-backed evaluator episode has completed")
    elif status == "exact_checkpoint_ready_pending_model_probe":
        if identifier != "univla_baai":
            issue(issues, f"closed loop: unsupported exact-checkpoint readiness status for {identifier}")
        evidence_keys = (
            ("checkpoint_receipt_path", "checkpoint_receipt_sha256"),
            ("paired_preflight_path", "paired_preflight_sha256"),
            ("paired_wrapper_path", "paired_wrapper_sha256"),
            ("adapter_path", "adapter_sha256"),
            ("evaluator_task_manifest_path", "evaluator_task_manifest_sha256"),
            ("official_evaluator_source_path", "official_evaluator_source_sha256"),
            ("official_policy_source_path", "official_policy_source_sha256"),
            ("superseded_blocker_path", "superseded_blocker_sha256"),
        )
        for path_key, hash_key in evidence_keys:
            evidence_path = Path(record.get(path_key, ""))
            if not evidence_path.is_file() or sha256(evidence_path) != record.get(hash_key):
                issue(issues, f"closed loop: UniVLA readiness evidence differs: {path_key}")
        expected_files = {
            "policy/model-00001-of-00004.safetensors": (
                4_884_766_656, "b7e6e472d974d1c572138839008e308486b526ffe1c9065306bd2107513b9e80",
                "df6c94ff872f4fc056693e507c033ca390694118",
            ),
            "policy/model-00002-of-00004.safetensors": (
                4_999_819_320, "c700932f380da894537a191b1b60f335a617efa01dc44ae6d8216a4f4e97f28d",
                "df6c94ff872f4fc056693e507c033ca390694118",
            ),
            "policy/model-00003-of-00004.safetensors": (
                4_915_916_184, "eeef8be71a34052ff2a590a2120f5d4067114f4f2f72b3c9b78d771968b207c3",
                "df6c94ff872f4fc056693e507c033ca390694118",
            ),
            "policy/model-00004-of-00004.safetensors": (
                2_183_554_760, "0932b91cf0b322993213a8fc5f0eb591171a130addd93d2bff701a3dc5e4259f",
                "df6c94ff872f4fc056693e507c033ca390694118",
            ),
            "vision/model.safetensors": (
                1_083_015_124, "89536431c69b08b10b449ec309f52dcea22f14b7647317f30f5715273392bbf1",
                "c81f916ad371289e205310a7539255e8a9396488",
            ),
        }
        receipt = read_json(Path(record.get("checkpoint_receipt_path", ""))) or {}
        files = receipt.get("files") or {}
        required_seals = (
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        )
        files_valid = set(files) == set(expected_files)
        if files_valid:
            for key, (size, digest, revision) in expected_files.items():
                row = files.get(key) or {}
                artifact = Path(row.get("path", ""))
                if (
                    row.get("bytes") != size
                    or row.get("sha256") != digest
                    or row.get("revision") != revision
                    or row.get("verification")
                        != "full_stream_sha256_then_memfd_sealed_immutable"
                    or not artifact.is_file()
                    or artifact.stat().st_size != size
                ):
                    files_valid = False
                    break
                try:
                    with artifact.open("rb") as stream:
                        seals = fcntl.fcntl(stream.fileno(), fcntl.F_GET_SEALS)
                except OSError:
                    files_valid = False
                    break
                if seals & required_seals != required_seals:
                    files_valid = False
                    break
        if (
            record.get("checkpoint_availability") != "resident_sealed_memfd"
            or receipt.get("all_pinned_files_verified") is not True
            or receipt.get("storage") != "sealed_memfd"
            or not files_valid
        ):
            issue(issues, "closed loop: UniVLA exact sealed checkpoint readiness differs")
        paired_preflight = read_json(Path(record.get("paired_preflight_path", ""))) or {}
        if (
            paired_preflight.get("schema_version") != 1
            or paired_preflight.get("model_id") != "univla_baai"
            or paired_preflight.get("passed") is not True
            or paired_preflight.get("checkpoint_receipt_sha256")
                != record.get("checkpoint_receipt_sha256")
            or paired_preflight.get("task_manifest_sha256")
                != record.get("evaluator_task_manifest_sha256")
            or paired_preflight.get("paired_wrapper_sha256")
                != record.get("paired_wrapper_sha256")
            or paired_preflight.get("adapter_sha256") != record.get("adapter_sha256")
            or paired_preflight.get("official_univla_revision")
                != "a91ee5f269d7ad46d46635ee6481d2eb655b4af9"
            or paired_preflight.get("official_lerobot_revision")
                != "e595b7902714ba51f91e47523f66f89c5181b649"
        ):
            issue(issues, "closed loop: UniVLA paired evaluator preflight differs")
        task_manifest = read_json(Path(record.get("evaluator_task_manifest_path", ""))) or {}
        tasks = task_manifest.get("tasks") or []
        if (
            task_manifest.get("model_id") != "univla_baai"
            or task_manifest.get("task_suite") != "libero_10"
            or task_manifest.get("task_ids") != list(range(10))
            or task_manifest.get("task_descriptions") != WSA_TASK_DESCRIPTIONS
            or task_manifest.get("initializations_per_task") != 50
            or len(tasks) != 10
            or any(
                row.get("task_id") != index
                or row.get("task_description") != WSA_TASK_DESCRIPTIONS[index]
                or row.get("initializations") != 50
                or not Path(row.get("init_states_path", "")).is_file()
                or sha256(Path(row.get("init_states_path", "")))
                    != row.get("init_states_sha256")
                or not Path(row.get("bddl_path", "")).is_file()
                or sha256(Path(row.get("bddl_path", ""))) != row.get("bddl_sha256")
                for index, row in enumerate(tasks)
            )
        ):
            issue(issues, "closed loop: UniVLA evaluator task manifest differs")
        if (
            record.get("official_univla_revision")
                != "a91ee5f269d7ad46d46635ee6481d2eb655b4af9"
            or record.get("official_lerobot_revision")
                != "e595b7902714ba51f91e47523f66f89c5181b649"
            or record.get("policy_checkpoint_revision")
                != "df6c94ff872f4fc056693e507c033ca390694118"
            or record.get("vision_checkpoint_revision")
                != "c81f916ad371289e205310a7539255e8a9396488"
            or record.get("deterministic_equivalent") is not True
            or record.get("deterministic_contract_sha256")
                != "944a2294c3a2924b64298a77aaa45babe4e98bf47652b771b409265c716028d8"
        ):
            issue(issues, "closed loop: UniVLA exact deterministic evaluator contract differs")
        if (record.get("probe_episodes_completed") != 0
                or record.get("paired_episodes_completed") != 0):
            issue(issues, "closed loop: UniVLA readiness record invents model episodes")
        issue(issues, "closed loop: UniVLA exact checkpoint/evaluator preflight passes, but no model-backed episode has completed")
    elif status == "blocked":
        command = record.get("attempted_command")
        stderr = record.get("stderr_path")
        stdout = record.get("stdout_path")
        if not isinstance(command, list) or not command or not record.get("attempt_timestamp_utc") or not isinstance(record.get("exit_code"), int) or record["exit_code"] == 0:
            issue(issues, f"closed loop: no failed evaluator command recorded for {identifier}")
        if not stderr or not Path(stderr).is_file() or not stdout or not Path(stdout).is_file():
            issue(issues, f"closed loop: no model-specific evaluator stdout/stderr files for {identifier}")
        elif not (Path(stderr).read_text().strip() or Path(stdout).read_text().strip()):
            issue(issues, f"closed loop: evaluator attempt has no captured failure output for {identifier}")
        if not record.get("failure_class") or not record.get("evaluator"):
            issue(issues, f"closed loop: blocker classification/evaluator missing for {identifier}")
        if record.get("failure_class") in {
            "missing_evaluator_dependency", "missing_evaluator_asset",
            "evaluator_startup_timeout", "evaluator_rollout_timeout",
        }:
            issue(issues, f"closed loop: {identifier} has a remediable setup/timeout failure, not a proven evaluator exception")
        investigation = record.get("blocker_investigation") or {}
        if not isinstance(investigation, dict):
            investigation = {}
        if not investigation.get("reason_permanent") or not investigation.get("resolution_attempts"):
            issue(issues, f"closed loop: {identifier} blocker lacks documented remediation/probing evidence")
        evidence = investigation.get("evidence_files") or []
        if not evidence or any(
            not isinstance(item, dict) or not Path(item.get("path", "")).is_file()
            or sha256(Path(item["path"])) != item.get("sha256")
            for item in evidence
        ):
            issue(issues, f"closed loop: {identifier} blocker lacks pinned investigation files")
        checkpoint_weight = Path(record.get("checkpoint_weight_path", ""))
        if not checkpoint_weight.is_file() or sha256(checkpoint_weight) != record.get("checkpoint_weight_sha256"):
            issue(issues, f"closed loop: evaluator blocker is not tied to resident exact checkpoint for {identifier}")
        source_evidence = Path(record.get("source_evidence_path", ""))
        if not source_evidence.is_file() or sha256(source_evidence) != record.get("source_evidence_sha256"):
            issue(issues, f"closed loop: evaluator blocker lacks pinned official source for {identifier}")
    elif status == "completed":
        episodes_path = record.get("episodes_path")
        if not episodes_path or not Path(episodes_path).is_file():
            issue(issues, f"closed loop: completed claim lacks raw episodes for {identifier}")
        else:
            if identifier == "openhelix":
                openhelix_evaluator_evidence_audit(record, Path(episodes_path), issues)
                return {"complete": len(issues) == start, "status": status}
            if identifier in {"spatialvla", "cogact_small"}:
                episodes_file = Path(episodes_path)
                if sha256(episodes_file) != record.get("episodes_sha256"):
                    issue(issues, f"closed loop: {identifier} episode stream hash differs")
                source_evidence = Path(record.get("source_evidence_path", ""))
                evaluator_source = Path(record.get("official_evaluator_source_path", ""))
                if (not source_evidence.is_file()
                        or sha256(source_evidence) != record.get("source_evidence_sha256")):
                    issue(issues, f"closed loop: {identifier} lacks pinned checkpoint/evaluator evidence")
                if (not evaluator_source.is_file()
                        or sha256(evaluator_source) != record.get("official_evaluator_source_sha256")):
                    issue(issues, f"closed loop: {identifier} official evaluator source differs")
                if identifier == "spatialvla":
                    spatialvla_evaluator_evidence_audit(record, episodes_file, issues)
                else:
                    cogact_evaluator_evidence_audit(record, episodes_file, issues)
                return {"complete": len(issues) == start, "status": status}
            episodes = list(stream_jsonl(Path(episodes_path)))
            condition_order = (
                "native_singleton", "native_dynamic",
                "patched_singleton", "patched_dynamic",
            )
            conditions = set(condition_order)
            if identifier in ("smolvla", "xvla", "pi_fast", "molmoact2", "univla_baai"):
                expected_tasks = set(range(10))
            else:
                task_manifest = record.get("evaluator_task_manifest_path")
                if (not task_manifest or not Path(task_manifest).is_file()
                        or sha256(Path(task_manifest)) != record.get("evaluator_task_manifest_sha256")):
                    issue(issues, f"closed loop: {identifier} lacks a pinned evaluator task manifest")
                    expected_tasks = set()
                else:
                    expected_tasks = set((read_json(Path(task_manifest)) or {}).get("task_ids", []))
                    if not expected_tasks:
                        issue(issues, f"closed loop: {identifier} evaluator task manifest is empty")
            start_seed = record.get("paired_start_seed")
            if not isinstance(start_seed, int):
                issue(issues, f"closed loop: {identifier} lacks a paired reset-seed schedule")
            checkpoint_weight = Path(record.get("checkpoint_weight_path", ""))
            if not checkpoint_weight.is_file() or sha256(checkpoint_weight) != record.get("checkpoint_weight_sha256"):
                issue(issues, f"closed loop: {identifier} completed result lacks exact checkpoint provenance")
            source_evidence = Path(record.get("source_evidence_path", ""))
            if not source_evidence.is_file() or sha256(source_evidence) != record.get("source_evidence_sha256"):
                issue(issues, f"closed loop: {identifier} completed result lacks pinned official evaluator source")
            groups = defaultdict(dict)
            for episode in episodes:
                task = episode.get("task")
                initialization = episode.get("initialization_id")
                condition = episode.get("condition")
                if task not in expected_tasks or not isinstance(initialization, int) or not 0 <= initialization < 50 or condition not in conditions:
                    issue(issues, f"closed loop: unexpected task/reset/condition for {identifier}")
                    break
                key = (task, initialization)
                if condition in groups[key]:
                    issue(issues, f"closed loop: duplicate paired episode {key} {condition}")
                groups[key][condition] = episode
                if (not isinstance(episode.get("success"), bool)
                        or not isinstance(episode.get("trajectory"), list) or not episode["trajectory"]
                        or not isinstance(episode.get("action_sequence"), list) or not episode["action_sequence"]
                        or not is_sha256_hex(episode.get("initial_observation_sha256"))
                        or (identifier in ("smolvla", "xvla", "pi_fast", "molmoact2", "univla_baai")
                            and not is_sha256_hex(episode.get("initial_raw_observation_sha256")))
                        or not is_sha256_hex(episode.get("first_noise_sha256"))
                        or not isinstance(start_seed, int)
                        or episode.get("reset_seed") != start_seed + initialization
                        or episode.get("explicit_init_state_id") != initialization):
                    issue(issues, f"closed loop: episode outcome/trajectory/reset evidence missing for {identifier}")
                    break
            expected_pairs = {(task, initialization) for task in expected_tasks for initialization in range(50)}
            if set(groups) != expected_pairs or any(set(group) != conditions for group in groups.values()):
                issue(issues, f"closed loop: {identifier} lacks four paired conditions for every task and 50 resets")
            if len(episodes) != 4 * len(expected_pairs):
                issue(issues, f"closed loop: {identifier} has duplicate or missing paired episode records")
            cross_implementation_reset_mismatches = []
            for key, group in groups.items():
                sensors = {
                    condition: group.get(condition, {}).get("initial_observation_sha256")
                    for condition in condition_order
                }
                if (sensors["native_singleton"] != sensors["native_dynamic"]
                        or sensors["patched_singleton"] != sensors["patched_dynamic"]):
                    issue(
                        issues,
                        f"closed loop: {identifier} within-implementation reset observation differs at {key}",
                    )
                    break
                if len(set(sensors.values())) != 1:
                    cross_implementation_reset_mismatches.append({
                        "task": key[0], "initialization_id": key[1], "hashes": sensors,
                    })
                if identifier in ("smolvla", "xvla", "pi_fast", "molmoact2", "univla_baai"):
                    raw_sensors = {
                        item.get("initial_raw_observation_sha256") for item in group.values()
                    }
                    if (len(raw_sensors) != 1
                            or not all(is_sha256_hex(value) for value in raw_sensors)):
                        issue(issues, f"closed loop: {identifier} raw reset observation differs at {key}")
                        break
                if len({item.get("first_noise_sha256") for item in group.values()}) != 1:
                    issue(issues, f"closed loop: {identifier} request-local noise differs across conditions at {key}")
                    break
                if identifier == "pi_fast":
                    deterministic_contract = {
                        "temperature": 0.0, "use_kv_cache": True,
                        "max_decoding_steps": 256,
                        "stochastic_request_inputs": "none_greedy_argmax",
                    }
                    deterministic_hash = hashlib.sha256(
                        json.dumps(deterministic_contract, sort_keys=True).encode()
                    ).hexdigest()
                    if any(item.get("first_noise_sha256") != deterministic_hash for item in group.values()):
                        issue(issues, f"closed loop: pi_fast deterministic contract differs at {key}")
                        break
                if identifier == "openvla_oft":
                    deterministic_contract = {
                        "inference_head": "l1_regression",
                        "stochastic_request_inputs": "none",
                        "do_sample": False,
                    }
                    deterministic_hash = hashlib.sha256(
                        json.dumps(deterministic_contract, sort_keys=True).encode()
                    ).hexdigest()
                    if any(item.get("first_noise_sha256") != deterministic_hash for item in group.values()):
                        issue(issues, f"closed loop: openvla_oft deterministic contract differs at {key}")
                        break
                if identifier == "univla_baai":
                    deterministic_contract = {
                        "action_decoding": "greedy_constrained_FAST",
                        "do_sample": False,
                        "eoa_token_id": 151845,
                        "maximum_new_tokens": 80,
                    }
                    deterministic_hash = hashlib.sha256(
                        json.dumps(
                            deterministic_contract, sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    if any(
                        item.get("first_noise_sha256") != deterministic_hash
                        for item in group.values()
                    ):
                        issue(issues, f"closed loop: univla_baai deterministic contract differs at {key}")
                        break
            official = record.get("condition_eval_info") or {}
            manifest_sources = (read_json(Path(record.get("paired_summary_path", ""))) or {}).get("condition_manifests", {}) if record.get("paired_summary_path") and Path(record["paired_summary_path"]).is_file() else {}
            if identifier in ("smolvla", "xvla", "pi_fast", "molmoact2", "univla_baai"):
                if set(manifest_sources) != conditions:
                    issue(issues, f"closed loop: {identifier} lacks four pinned evaluator condition manifests")
                wrapper_path = Path(__file__).with_name(
                    "run_lerobot_pi0fast_paired_libero.py" if identifier == "pi_fast"
                    else "run_lerobot_smolvla_paired_libero.py" if identifier == "smolvla"
                    else "run_molmoact2_paired_libero.py" if identifier == "molmoact2"
                    else "run_univla_paired_libero.py" if identifier == "univla_baai"
                    else "run_lerobot_paired_libero.py"
                )
                wrapper_sha = sha256(wrapper_path) if wrapper_path.is_file() else None
                shared_wrapper_path = Path(__file__).with_name("run_lerobot_paired_libero.py")
                shared_wrapper_sha = sha256(shared_wrapper_path) if shared_wrapper_path.is_file() else None
                evaluator_source = Path(record.get("official_evaluator_source_path", ""))
                if (not evaluator_source.is_file()
                        or sha256(evaluator_source) != record.get("official_evaluator_source_sha256")):
                    issue(issues, f"closed loop: {identifier} lacks pinned official evaluator implementation")
                policy_dir = "pi0_fast" if identifier == "pi_fast" else identifier
                policy_source = (
                    Path(record.get("checkpoint_model_source_path", ""))
                    if identifier in ("molmoact2", "univla_baai")
                    else evaluator_source.parent.parent / "policies" / policy_dir /
                    f"modeling_{policy_dir}.py"
                )
                policy_sha = sha256(policy_source) if policy_source.is_file() else None
                tokenizer_report_path = path.parent.parent / "tokenizer_provenance.json"
                tokenizer_report_sha = (
                    sha256(tokenizer_report_path) if tokenizer_report_path.is_file() else None
                )
                if identifier == "pi_fast" and tokenizer_report_sha != PI_FAST_TOKENIZER_PROVENANCE_SHA256:
                    issue(issues, "closed loop: pi_fast lacks pinned byte-identical official tokenizer provenance")
                pi_fast_checkpoint_shape = None
                if identifier == "pi_fast" and checkpoint_weight.is_file():
                    from safetensors import safe_open

                    dtype_bytes = {"F32": 4, "BF16": 2, "F16": 2}
                    with safe_open(checkpoint_weight, framework="pt", device="cpu") as weights:
                        keys = list(weights.keys())
                        pi_fast_checkpoint_shape = (
                            len(keys),
                            sum(
                                int(np.prod(weights.get_slice(key).get_shape()))
                                * dtype_bytes[weights.get_slice(key).get_dtype()]
                                for key in keys
                            ),
                        )
                for condition in conditions & set(manifest_sources):
                    item = manifest_sources[condition]
                    manifest_path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
                    if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
                        issue(issues, f"closed loop: {identifier} condition manifest missing or hash-changed for {condition}")
                        continue
                    manifest = read_json(manifest_path) or {}
                    required_batch = 1 if condition.endswith("singleton") else 4
                    paired_protocol_version = manifest.get("paired_protocol_version", 1)
                    if paired_protocol_version == 1:
                        physical_batch_size = required_batch
                    elif paired_protocol_version == 2:
                        physical_batch_size = manifest.get("environment_batch_size")
                        expected_execution_mode = (
                            "serial_singleton" if required_batch == 1 else "dynamic_batch"
                        )
                        if (manifest.get("policy_batch_size") != required_batch
                                or physical_batch_size != 4
                                or manifest.get("execution_mode") != expected_execution_mode
                                or manifest.get("smoke_test") is not False):
                            issue(
                                issues,
                                f"closed loop: {identifier} protocol-v2 physical/policy batch "
                                f"isolation differs for {condition}",
                            )
                    else:
                        physical_batch_size = None
                        issue(
                            issues,
                            f"closed loop: {identifier} has unknown paired protocol version "
                            f"for {condition}",
                        )
                    precision = manifest.get("precision_controls") or {}
                    required_precision = {
                        "tf32_matmul": True, "tf32_cudnn": True,
                        "cudnn_benchmark": True, "cudnn_deterministic": False,
                        "deterministic_algorithms": False, "matmul_precision": "high",
                    }
                    if (manifest.get("model") != identifier
                            or manifest.get("condition") != condition
                            or manifest.get("batch_size") != required_batch
                            or manifest.get("invariant_ops") != condition.startswith("patched")
                            or manifest.get("explicit_init_state_by_initialization_id") is not True
                            or manifest.get("episodes_per_task") != 50
                            or manifest.get("seed") != start_seed
                            or manifest.get("checkpoint_sha256") != record.get("checkpoint_weight_sha256")
                            or manifest.get("checkpoint_weight") != str(checkpoint_weight.resolve())
                            or manifest.get("official_evaluator_sha256") != record.get("official_evaluator_source_sha256")
                            or manifest.get("official_policy_sha256") != policy_sha
                            or manifest.get("paired_wrapper_sha256") != wrapper_sha
                            or any(precision.get(key) != value for key, value in required_precision.items())
                            or not precision.get("torch") or not precision.get("cuda")):
                        issue(issues, f"closed loop: {identifier} condition manifest provenance differs for {condition}")
                    if manifest.get("task_suite") != "libero_10":
                        issue(issues, f"closed loop: {identifier} task suite differs for {condition}")
                    if identifier == "xvla" and (
                        manifest.get("control_mode") != "absolute"
                        or manifest.get("episode_length") != 800
                    ):
                        issue(
                            issues,
                            f"closed loop: xvla official absolute-control/800-step recipe differs for {condition}",
                        )
                    if identifier == "pi_fast" and (
                        manifest.get("control_mode") != "relative"
                        or manifest.get("episode_length") != 520
                    ):
                        issue(
                            issues,
                            f"closed loop: pi_fast relative-control/520-step recipe differs for {condition}",
                        )
                    if identifier == "smolvla" and (
                        manifest.get("control_mode") != "relative"
                        or manifest.get("episode_length") != 520
                        or manifest.get("shared_rollout_wrapper_sha256") != shared_wrapper_sha
                    ):
                        issue(
                            issues,
                            f"closed loop: smolvla relative-control/520-step recipe differs for {condition}",
                        )
                    if identifier == "molmoact2":
                        task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
                        paired_preflight_path = path.parent / "paired_preflight/exact_ram_stream_v3/preflight.json"
                        if (manifest.get("control_mode") != "relative"
                                or manifest.get("episode_length") != 520
                                or manifest.get("request_local_noise") is not True
                                or manifest.get("checkpoint_loading")
                                    != "five_hash_verified_shards_streamed_from_RAM_to_meta_allocated_GPU_model"
                                or manifest.get("checkpoint_revision")
                                    != "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
                                or manifest.get("official_model_repository_revision")
                                    != "66b87e64efd99dfd103241418113955cf64dfa9c"
                                or manifest.get("official_lerobot_revision")
                                    != "e595b7902714ba51f91e47523f66f89c5181b649"
                                or manifest.get("shared_rollout_wrapper_sha256") != shared_wrapper_sha
                                or manifest.get("task_manifest_path")
                                    != str(task_manifest_path.resolve())
                                or manifest.get("task_manifest_sha256")
                                    != record.get("evaluator_task_manifest_sha256")
                                or manifest.get("paired_preflight_path")
                                    != str(paired_preflight_path.resolve())
                                or not paired_preflight_path.is_file()
                                or manifest.get("paired_preflight_sha256")
                                    != sha256(paired_preflight_path)):
                            issue(
                                issues,
                                f"closed loop: molmoact2 exact stream/evaluator recipe differs for {condition}",
                            )
                    if identifier == "univla_baai":
                        task_manifest_path = Path(
                            record.get("evaluator_task_manifest_path", "")
                        )
                        deterministic_contract = {
                            "action_decoding": "greedy_constrained_FAST",
                            "do_sample": False,
                            "eoa_token_id": 151845,
                            "maximum_new_tokens": 80,
                        }
                        deterministic_hash = hashlib.sha256(
                            json.dumps(
                                deterministic_contract, sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                        if (
                            manifest.get("control_mode") != "relative"
                            or manifest.get("episode_length") != 520
                            or manifest.get("request_local_noise") is not False
                            or manifest.get("deterministic_equivalent") is not True
                            or manifest.get("deterministic_contract_sha256")
                                != deterministic_hash
                            or manifest.get("checkpoint_loading")
                                != "five_hash_verified_sealed_memfd_artifacts"
                            or manifest.get("official_univla_revision")
                                != "a91ee5f269d7ad46d46635ee6481d2eb655b4af9"
                            or manifest.get("official_lerobot_revision")
                                != "e595b7902714ba51f91e47523f66f89c5181b649"
                            or manifest.get("policy_checkpoint_revision")
                                != "df6c94ff872f4fc056693e507c033ca390694118"
                            or manifest.get("vision_checkpoint_revision")
                                != "c81f916ad371289e205310a7539255e8a9396488"
                            or manifest.get("shared_rollout_wrapper_sha256")
                                != shared_wrapper_sha
                            or manifest.get("task_manifest_path")
                                != str(task_manifest_path.resolve())
                            or manifest.get("task_manifest_sha256")
                                != record.get("evaluator_task_manifest_sha256")
                        ):
                            issue(
                                issues,
                                f"closed loop: univla_baai exact sealed-checkpoint/evaluator recipe differs for {condition}",
                            )
                    if identifier == "pi_fast":
                        expected_contract = {
                            "temperature": 0.0, "use_kv_cache": True,
                            "max_decoding_steps": 256,
                            "stochastic_request_inputs": "none_greedy_argmax",
                        }
                        expected_contract_sha = hashlib.sha256(
                            json.dumps(expected_contract, sort_keys=True).encode()
                        ).hexdigest()
                        if (manifest.get("shared_rollout_wrapper_sha256") != shared_wrapper_sha
                                or manifest.get("tokenizer_provenance_sha256") != tokenizer_report_sha
                                or manifest.get("deterministic_decoding_contract") != expected_contract
                                or manifest.get("deterministic_decoding_sha256") != expected_contract_sha):
                            issue(issues, f"closed loop: pi_fast deterministic/tokenizer provenance differs for {condition}")
                        verification = read_json(manifest_path.parent / "checkpoint_load_verification.json") or {}
                        if (pi_fast_checkpoint_shape is None
                                or verification.get("checkpoint_sha256") != record.get("checkpoint_weight_sha256")
                                or (verification.get("tensors_checked"), verification.get("source_bytes_checked")) != pi_fast_checkpoint_shape
                                or verification.get("exact_after_policy_dtype_conversion") is not True):
                            issue(issues, f"closed loop: pi_fast official checkpoint load unverified for {condition}")
                    if identifier == "smolvla":
                        verification = read_json(manifest_path.parent / "checkpoint_load_verification.json") or {}
                        if (verification.get("checkpoint_sha256") != record.get("checkpoint_weight_sha256")
                                or verification.get("tensors_checked") != 788
                                or verification.get("source_bytes_checked") != 1_217_932_096
                                or verification.get("exact_after_policy_dtype_conversion") is not True):
                            issue(
                                issues,
                                f"closed loop: smolvla official checkpoint load unverified for {condition}",
                            )
                    expected_info = manifest_path.parent / "official" / "eval_info.json"
                    source = official.get(condition, {})
                    if not isinstance(source, dict) or source.get("path") != str(expected_info.resolve()):
                        issue(issues, f"closed loop: {identifier} official metrics are not tied to the {condition} condition")

                    # Reconstruct the physical evaluator requests directly from the
                    # per-batch rollout files.  The paired JSONL is a derived artifact;
                    # it must not be able to manufacture reset provenance that is absent
                    # from (or disagrees with) the raw official-evaluator capture.
                    rollout_paths = rollout_json_paths(manifest_path.parent / "rollouts")
                    raw_rows = {}
                    valid_physical_batch_size = physical_batch_size in (1, 4)
                    audit_physical_batch_size = (
                        physical_batch_size if valid_physical_batch_size else required_batch
                    )
                    expected_physical_initializations = range(
                        50 if audit_physical_batch_size == 1 else 52
                    )
                    expected_raw_keys = {
                        (task, initialization)
                        for task in expected_tasks
                        for initialization in expected_physical_initializations
                    }
                    expected_file_count = len(expected_raw_keys) // audit_physical_batch_size
                    if len(rollout_paths) != expected_file_count:
                        issue(
                            issues,
                            f"closed loop: {identifier} {condition} has "
                            f"{len(rollout_paths)}/{expected_file_count} physical rollout batches",
                        )
                    for rollout_path in rollout_paths:
                        payload = read_json(rollout_path) or {}
                        task = payload.get("task_id")
                        seeds = payload.get("seeds")
                        explicit_ids = payload.get("explicit_init_state_ids")
                        raw_episodes = payload.get("episodes")
                        tensor_path = (
                            rollout_path.with_suffix("").with_suffix(".pt")
                            if rollout_path.suffix == ".gz"
                            else rollout_path.with_suffix(".pt")
                        )
                        if not tensor_path.is_file():
                            issue(
                                issues,
                                f"closed loop: {identifier} physical rollout tensor missing "
                                f"for {rollout_path}",
                            )
                        if (not valid_physical_batch_size
                                or task not in expected_tasks
                                or not isinstance(seeds, list)
                                or not isinstance(explicit_ids, list)
                                or not isinstance(raw_episodes, list)
                                or len(seeds) != audit_physical_batch_size
                                or len(explicit_ids) != audit_physical_batch_size
                                or len(raw_episodes) != audit_physical_batch_size
                                or not isinstance(start_seed, int)
                                or any(not isinstance(seed, int) for seed in seeds)
                                or explicit_ids != [seed - start_seed for seed in seeds]
                                or (paired_protocol_version == 2
                                    and payload.get("task_group") != "libero_10")):
                            issue(issues, f"closed loop: {identifier} malformed physical rollout batch {rollout_path}")
                            continue
                        seed_to_initialization = dict(zip(seeds, explicit_ids))
                        if len(seed_to_initialization) != audit_physical_batch_size:
                            issue(issues, f"closed loop: {identifier} duplicate reset seed in {rollout_path}")
                            continue
                        seen_in_file = set()
                        for raw_episode in raw_episodes:
                            if not isinstance(raw_episode, dict):
                                issue(issues, f"closed loop: {identifier} non-object episode in {rollout_path}")
                                continue
                            reset_seed = raw_episode.get("reset_seed")
                            initialization = raw_episode.get("initialization_id")
                            raw_key = (task, initialization)
                            if (raw_episode.get("task") != task
                                    or (paired_protocol_version == 2
                                        and raw_episode.get("task_group") != "libero_10")
                                    or reset_seed not in seed_to_initialization
                                    or seed_to_initialization.get(reset_seed) != initialization
                                    or initialization not in expected_physical_initializations):
                                issue(issues, f"closed loop: {identifier} raw reset assignment differs in {rollout_path}")
                                continue
                            if raw_key in raw_rows or initialization in seen_in_file:
                                issue(issues, f"closed loop: {identifier} duplicate raw reset {raw_key} for {condition}")
                                continue
                            seen_in_file.add(initialization)
                            raw_rows[raw_key] = (raw_episode, rollout_path)
                    if set(raw_rows) != expected_raw_keys:
                        issue(
                            issues,
                            f"closed loop: {identifier} {condition} raw physical reset coverage differs "
                            f"({len(raw_rows)}/{len(expected_raw_keys)})",
                        )
                    for raw_key in expected_pairs & set(raw_rows):
                        combined = groups.get(raw_key, {}).get(condition)
                        raw_episode, rollout_path = raw_rows[raw_key]
                        if combined is None:
                            continue
                        expected_combined = dict(raw_episode)
                        expected_combined.update({
                            "condition": condition,
                            "source_rollout_path": str(rollout_path.resolve()),
                            "explicit_init_state_id": raw_key[1],
                        })
                        if combined != expected_combined:
                            issue(
                                issues,
                                f"closed loop: {identifier} reduced episode differs from raw rollout "
                                f"for {condition} {raw_key}",
                            )
                            break
            if identifier == "molmoact2":
                expected_shards = (
                    ("model-00001-of-00005.safetensors", 4_919_324_120,
                     "5aa56113a3c1c95f0c03b90bada5b7ac2babf59c752a3d6201b8362a530c6809"),
                    ("model-00002-of-00005.safetensors", 4_844_690_992,
                     "1fd95aec9ad144d3f4eac42db88a000321e016beeae606ecfcb84fcbdb13ecf4"),
                    ("model-00003-of-00005.safetensors", 4_844_691_024,
                     "e6c455434915e74a8d750bedd0fc4c23b5a8ce37954aaff9c3e3f22085a26890"),
                    ("model-00004-of-00005.safetensors", 4_877_619_536,
                     "4e5bcf5bad81573153d6deaa3d25c3e53dc697658488b1cdbab4c8a3dffba1b6"),
                    ("model-00005-of-00005.safetensors", 2_282_630_240,
                     "76140ad00c225f95b80eb6b6a5fea2f985e4fdde3356a3c67c93d87d083d4ee2"),
                )
                receipt_path = Path(record.get("checkpoint_stream_receipt_path", ""))
                receipt = read_json(receipt_path) or {}
                strict = receipt.get("strict_model_load") or {}
                files = receipt.get("files") or {}
                if (not receipt_path.is_file()
                        or sha256(receipt_path)
                            != record.get("checkpoint_stream_receipt_sha256")
                        or receipt.get("repository") != "allenai/MolmoAct2-LIBERO"
                        or receipt.get("revision")
                            != "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
                        or receipt.get("all_pinned_files_verified") is not True
                        or set(files) != {row[0] for row in expected_shards}
                        or any(files.get(name, {}).get("bytes") != size
                               or files.get(name, {}).get("sha256") != digest
                               or files.get(name, {}).get("verification")
                                   != "full_file_sha256_before_tensor_load"
                               for name, size, digest in expected_shards)
                        or strict != {
                            "expected_state_tensors": 1295,
                            "loaded_state_tensors": 1295,
                            "missing_state_tensors": [],
                            "meta_parameters_remaining": [],
                            "meta_buffers_remaining": [],
                            "target_parameter_dtype": "torch.bfloat16",
                        }):
                    issue(issues, "closed loop: molmoact2 exact streamed checkpoint load differs")
                task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
                task_manifest = read_json(task_manifest_path) or {}
                task_rows = task_manifest.get("tasks") or []
                if (not task_manifest_path.is_file()
                        or sha256(task_manifest_path)
                            != record.get("evaluator_task_manifest_sha256")
                        or task_manifest.get("model_id") != "molmoact2"
                        or task_manifest.get("task_suite") != "libero_10"
                        or task_manifest.get("task_ids") != list(range(10))
                        or task_manifest.get("task_descriptions") != WSA_TASK_DESCRIPTIONS
                        or task_manifest.get("initializations_per_task") != 50
                        or task_manifest.get("checkpoint_repository_revision")
                            != "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
                        or task_manifest.get("checkpoint_index_sha256")
                            != "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06"
                        or task_manifest.get("checkpoint_shards") != [
                            {"name": name, "bytes": size, "sha256": digest}
                            for name, size, digest in expected_shards
                        ]
                        or len(task_rows) != 10
                        or any(row.get("task_id") != index
                               or row.get("task_description") != WSA_TASK_DESCRIPTIONS[index]
                               or row.get("initializations") != 50
                               or not Path(row.get("init_states_path", "")).is_file()
                               or sha256(Path(row.get("init_states_path", "")))
                                   != row.get("init_states_sha256")
                               or not Path(row.get("bddl_path", "")).is_file()
                               or sha256(Path(row.get("bddl_path", "")))
                                   != row.get("bddl_sha256")
                               for index, row in enumerate(task_rows))):
                    issue(issues, "closed loop: molmoact2 evaluator task/checkpoint manifest differs")
                model_source = Path(record.get("checkpoint_model_source_path", ""))
                if (not model_source.is_file()
                        or sha256(model_source) != record.get("checkpoint_model_source_sha256")
                        or record.get("official_model_repository_revision")
                            != "66b87e64efd99dfd103241418113955cf64dfa9c"
                        or record.get("official_lerobot_revision")
                            != "e595b7902714ba51f91e47523f66f89c5181b649"
                        or record.get("checkpoint_repository_revision")
                            != "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"):
                    issue(issues, "closed loop: molmoact2 official source revisions differ")
            if identifier == "univla_baai":
                expected_files = {
                    "policy/model-00001-of-00004.safetensors": (
                        4_884_766_656,
                        "b7e6e472d974d1c572138839008e308486b526ffe1c9065306bd2107513b9e80",
                        "df6c94ff872f4fc056693e507c033ca390694118",
                    ),
                    "policy/model-00002-of-00004.safetensors": (
                        4_999_819_320,
                        "c700932f380da894537a191b1b60f335a617efa01dc44ae6d8216a4f4e97f28d",
                        "df6c94ff872f4fc056693e507c033ca390694118",
                    ),
                    "policy/model-00003-of-00004.safetensors": (
                        4_915_916_184,
                        "eeef8be71a34052ff2a590a2120f5d4067114f4f2f72b3c9b78d771968b207c3",
                        "df6c94ff872f4fc056693e507c033ca390694118",
                    ),
                    "policy/model-00004-of-00004.safetensors": (
                        2_183_554_760,
                        "0932b91cf0b322993213a8fc5f0eb591171a130addd93d2bff701a3dc5e4259f",
                        "df6c94ff872f4fc056693e507c033ca390694118",
                    ),
                    "vision/model.safetensors": (
                        1_083_015_124,
                        "89536431c69b08b10b449ec309f52dcea22f14b7647317f30f5715273392bbf1",
                        "c81f916ad371289e205310a7539255e8a9396488",
                    ),
                }
                receipt_path = Path(record.get("checkpoint_receipt_path", ""))
                receipt = read_json(receipt_path) or {}
                files = receipt.get("files") or {}
                required_seals = (
                    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW
                    | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
                )
                receipt_files_valid = set(files) == set(expected_files)
                if receipt_files_valid:
                    for key, (size, digest, revision) in expected_files.items():
                        row = files.get(key) or {}
                        artifact = Path(row.get("path", ""))
                        if (
                            row.get("bytes") != size
                            or row.get("sha256") != digest
                            or row.get("revision") != revision
                            or row.get("verification")
                                != "full_stream_sha256_then_memfd_sealed_immutable"
                            or not artifact.is_file()
                            or artifact.stat().st_size != size
                        ):
                            receipt_files_valid = False
                            break
                        try:
                            with artifact.open("rb") as stream:
                                seals = fcntl.fcntl(stream.fileno(), fcntl.F_GET_SEALS)
                        except OSError:
                            receipt_files_valid = False
                            break
                        if seals & required_seals != required_seals:
                            receipt_files_valid = False
                            break
                if (
                    not receipt_path.is_file()
                    or sha256(receipt_path) != record.get("checkpoint_receipt_sha256")
                    or receipt.get("all_pinned_files_verified") is not True
                    or receipt.get("storage") != "sealed_memfd"
                    or not receipt_files_valid
                ):
                    issue(issues, "closed loop: univla_baai exact sealed checkpoint load differs")
                task_manifest_path = Path(record.get("evaluator_task_manifest_path", ""))
                task_manifest = read_json(task_manifest_path) or {}
                task_rows = task_manifest.get("tasks") or []
                checkpoint_rows = task_manifest.get("checkpoint_files") or []
                expected_checkpoint_rows = [
                    {
                        "key": key,
                        "path": files.get(key, {}).get("path"),
                        "bytes": size,
                        "sha256": digest,
                        "repository": files.get(key, {}).get("repository"),
                        "revision": revision,
                    }
                    for key, (size, digest, revision) in sorted(expected_files.items())
                ]
                if (
                    not task_manifest_path.is_file()
                    or sha256(task_manifest_path)
                        != record.get("evaluator_task_manifest_sha256")
                    or task_manifest.get("model_id") != "univla_baai"
                    or task_manifest.get("task_suite") != "libero_10"
                    or task_manifest.get("task_ids") != list(range(10))
                    or task_manifest.get("task_descriptions") != WSA_TASK_DESCRIPTIONS
                    or task_manifest.get("initializations_per_task") != 50
                    or task_manifest.get("official_univla_revision")
                        != "a91ee5f269d7ad46d46635ee6481d2eb655b4af9"
                    or task_manifest.get("official_lerobot_revision")
                        != "e595b7902714ba51f91e47523f66f89c5181b649"
                    or task_manifest.get("checkpoint_receipt_sha256")
                        != record.get("checkpoint_receipt_sha256")
                    or checkpoint_rows != expected_checkpoint_rows
                    or len(task_rows) != 10
                    or any(
                        row.get("task_id") != index
                        or row.get("task_description") != WSA_TASK_DESCRIPTIONS[index]
                        or row.get("initializations") != 50
                        or not Path(row.get("init_states_path", "")).is_file()
                        or sha256(Path(row.get("init_states_path", "")))
                            != row.get("init_states_sha256")
                        or not Path(row.get("bddl_path", "")).is_file()
                        or sha256(Path(row.get("bddl_path", "")))
                            != row.get("bddl_sha256")
                        for index, row in enumerate(task_rows)
                    )
                ):
                    issue(issues, "closed loop: univla_baai evaluator task/checkpoint manifest differs")
                model_source = Path(record.get("checkpoint_model_source_path", ""))
                if (
                    not model_source.is_file()
                    or sha256(model_source) != record.get("checkpoint_model_source_sha256")
                    or record.get("official_model_repository_revision")
                        != "a91ee5f269d7ad46d46635ee6481d2eb655b4af9"
                    or record.get("official_lerobot_revision")
                        != "e595b7902714ba51f91e47523f66f89c5181b649"
                    or record.get("policy_checkpoint_revision")
                        != "df6c94ff872f4fc056693e507c033ca390694118"
                    or record.get("vision_checkpoint_revision")
                        != "c81f916ad371289e205310a7539255e8a9396488"
                ):
                    issue(issues, "closed loop: univla_baai official source revisions differ")
            if identifier in ("internvla_a15_standard", "internvla_a15_optimized"):
                if set(manifest_sources) != conditions:
                    issue(issues, f"closed loop: {identifier} lacks four pinned evaluator condition manifests")
                backend_name = identifier.rsplit("_", 1)[-1]
                wrapper_dir = Path(__file__).parent
                client_wrapper = wrapper_dir / "run_internvla_paired_client.py"
                server_wrapper = wrapper_dir / "run_internvla_paired_server.py"
                client_wrapper_sha = sha256(client_wrapper) if client_wrapper.is_file() else None
                server_wrapper_sha = sha256(server_wrapper) if server_wrapper.is_file() else None
                evaluator_source = Path(record.get("official_evaluator_source_path", ""))
                if (not evaluator_source.is_file()
                        or sha256(evaluator_source) != record.get("official_evaluator_source_sha256")):
                    issue(issues, f"closed loop: {identifier} lacks pinned official evaluator implementation")
                    model_root = Path("")
                else:
                    model_root = evaluator_source.parents[2]
                backend_source = model_root / "evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py"
                policy_source = model_root / "src/lerobot/policies/internvla_a1_5" / (
                    "modeling_internvla_a1_5_optimized.py"
                    if backend_name == "optimized" else "modeling_internvla_a1_5.py"
                )
                backend_sha = sha256(backend_source) if backend_source.is_file() else None
                policy_sha = sha256(policy_source) if policy_source.is_file() else None
                vlm_weight = Path(record.get("vlm_weight_path", ""))
                if (not vlm_weight.is_file()
                        or sha256(vlm_weight) != record.get("vlm_weight_sha256")):
                    issue(issues, f"closed loop: {identifier} lacks exact VLM checkpoint provenance")
                for condition in conditions & set(manifest_sources):
                    item = manifest_sources[condition]
                    manifest_path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
                    if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
                        issue(issues, f"closed loop: {identifier} condition manifest missing or hash-changed for {condition}")
                        continue
                    manifest = read_json(manifest_path) or {}
                    required_batch = 1 if condition.endswith("singleton") else 4
                    required_padding = "fixed_right_650" if condition.startswith("patched") else "longest_right"
                    expected_server_wrapper_sha = (
                        INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256
                        if identifier == "internvla_a15_standard" and condition == "native_singleton"
                        else server_wrapper_sha
                    )
                    if (manifest.get("model") != identifier
                            or manifest.get("backend") != backend_name
                            or manifest.get("condition") != condition
                            or manifest.get("batch_size") != required_batch
                            or manifest.get("invariant_ops") != condition.startswith("patched")
                            or manifest.get("episodes_per_task") != 50
                            or manifest.get("seed") != start_seed
                            or manifest.get("task_suite") != "libero_10"
                            or manifest.get("episode_length") != 520
                            or manifest.get("num_steps_wait") != 10
                            or manifest.get("replan_steps") != 8
                            or manifest.get("explicit_init_state_by_initialization_id") is not True
                            or manifest.get("request_local_noise") is not True
                            or manifest.get("genuine_batched_policy_call") is not True
                            or manifest.get("prompt_padding") != required_padding
                            or manifest.get("checkpoint_sha256") != record.get("checkpoint_weight_sha256")
                            or manifest.get("checkpoint_weight") != str(checkpoint_weight.resolve())
                            or manifest.get("vlm_sha256") != record.get("vlm_weight_sha256")
                            or manifest.get("vlm_weight") != str(vlm_weight.resolve())
                            or manifest.get("official_evaluator_sha256") != record.get("official_evaluator_source_sha256")
                            or manifest.get("official_backend_sha256") != backend_sha
                            or manifest.get("official_policy_sha256") != policy_sha
                            or manifest.get("paired_wrapper_sha256") != client_wrapper_sha
                            or manifest.get("shared_rollout_wrapper_sha256") != expected_server_wrapper_sha
                            or manifest.get("optimized_graph_cache_policy")
                            != ("recapture_per_policy_call" if backend_name == "optimized" else None)):
                        issue(issues, f"closed loop: {identifier} paired provenance differs for {condition}")

                    server_metadata = read_json(manifest_path.parent / "server_metadata.json") or {}
                    if (server_metadata.get("policy_type") != "internvla_a1_5"
                            or server_metadata.get("ckpt_path") != str(checkpoint_weight.parent.resolve())
                            or server_metadata.get("paired_condition") != condition
                            or server_metadata.get("genuine_batched_policy_call") is not True
                            or server_metadata.get("maximum_request_batch") != required_batch
                            or server_metadata.get("batch_invariant_ops") != condition.startswith("patched")
                            or server_metadata.get("request_local_noise") is not True
                            or server_metadata.get("prompt_padding") != required_padding
                            or server_metadata.get("optimized_graph_cache_policy")
                            != ("recapture_per_policy_call" if backend_name == "optimized" else None)
                            or server_metadata.get("stats_key") != "libero_10"
                            or server_metadata.get("robot_type") != "panda"
                            or server_metadata.get("action_mode") != "joint"
                            or server_metadata.get("chunk_size") != 8
                            or server_metadata.get("action_dim") != 7
                            or server_metadata.get("protocol_version") != "2.1"
                            or server_metadata.get("preprocessing_owner") != "server_canonical"
                            or server_metadata.get("deterministic_inference_preprocess") is not True):
                        issue(issues, f"closed loop: {identifier} server handshake differs for {condition}")

                    paths = rollout_json_paths(manifest_path.parent / "rollouts")
                    expected_file_count = 500 if required_batch == 1 else 130
                    if len(paths) != expected_file_count:
                        issue(
                            issues,
                            f"closed loop: {identifier} {condition} has "
                            f"{len(paths)}/{expected_file_count} physical rollout groups",
                        )
                    expected_groups = {
                        (task, tuple(range(start, min(start + required_batch, 50))))
                        for task in expected_tasks for start in range(0, 50, required_batch)
                    }
                    observed_groups = set()
                    raw_rows = {}
                    for rollout_path in paths:
                        payload = read_json(rollout_path) or {}
                        task = payload.get("task_id")
                        seeds = payload.get("seeds")
                        explicit_ids = payload.get("explicit_init_state_ids")
                        raw_episodes = payload.get("episodes")
                        group_key = (task, tuple(explicit_ids) if isinstance(explicit_ids, list) else ())
                        if (group_key not in expected_groups or group_key in observed_groups
                                or not isinstance(seeds, list)
                                or not isinstance(raw_episodes, list)
                                or len(seeds) != len(group_key[1])
                                or len(raw_episodes) != len(group_key[1])
                                or not isinstance(start_seed, int)
                                or seeds != [start_seed + value for value in group_key[1]]):
                            issue(issues, f"closed loop: {identifier} malformed physical rollout group {rollout_path}")
                            continue
                        observed_groups.add(group_key)
                        for raw_episode in raw_episodes:
                            initialization = raw_episode.get("initialization_id")
                            raw_key = (task, initialization)
                            sizes = raw_episode.get("policy_batch_sizes")
                            if (raw_episode.get("task") != task
                                    or initialization not in group_key[1]
                                    or not isinstance(start_seed, int)
                                    or raw_episode.get("reset_seed") != start_seed + initialization
                                    or raw_key in raw_rows
                                    or not isinstance(sizes, list) or not sizes
                                    or sizes[0] != len(group_key[1])
                                    or any(not isinstance(size, int) or not 1 <= size <= len(group_key[1]) for size in sizes)):
                                issue(issues, f"closed loop: {identifier} raw batched request differs in {rollout_path}")
                                continue
                            raw_rows[raw_key] = (raw_episode, rollout_path)
                    if observed_groups != expected_groups:
                        issue(issues, f"closed loop: {identifier} {condition} physical group schedule is incomplete")
                    if set(raw_rows) != expected_pairs:
                        issue(issues, f"closed loop: {identifier} {condition} raw reset coverage differs")
                    for raw_key in expected_pairs & set(raw_rows):
                        combined = groups.get(raw_key, {}).get(condition)
                        raw_episode, rollout_path = raw_rows[raw_key]
                        if combined is None:
                            continue
                        expected_combined = dict(raw_episode)
                        expected_combined.update({
                            "condition": condition,
                            "source_rollout_path": str(rollout_path.resolve()),
                            "explicit_init_state_id": raw_key[1],
                        })
                        if combined != expected_combined:
                            issue(
                                issues,
                                f"closed loop: {identifier} reduced episode differs from raw rollout "
                                f"for {condition} {raw_key}",
                            )
                            break
            if identifier in ("uva", "uva_joint"):
                uva_evaluator_evidence_audit(
                    record, identifier, checkpoint_weight, manifest_sources, conditions,
                    expected_tasks, expected_pairs, groups, start_seed, issues,
                )
            if identifier == "openvla_oft":
                openvla_oft_evaluator_evidence_audit(
                    record, checkpoint_weight, manifest_sources, conditions,
                    expected_tasks, expected_pairs, groups, start_seed, issues,
                )
            if identifier == "wsa_base":
                wsa_evaluator_evidence_audit(
                    record, checkpoint_weight, manifest_sources, conditions,
                    expected_tasks, expected_pairs, groups, start_seed, issues,
                )
            if identifier == "rdt_1b":
                rdt_evaluator_evidence_audit(
                    record, checkpoint_weight, manifest_sources, conditions,
                    expected_tasks, expected_pairs, groups, start_seed, issues,
                )
            if set(official) != conditions:
                issue(issues, f"closed loop: {identifier} lacks four official evaluator metric files")
            elif len({official[condition].get("path") for condition in conditions if isinstance(official[condition], dict)}) != 4:
                issue(issues, f"closed loop: {identifier} reuses an official metric file across conditions")
            official_per_task = {}
            official_success_granularity = {}
            for condition in conditions & set(official):
                item = official[condition]
                info_path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
                if not info_path.is_file() or sha256(info_path) != item.get("sha256"):
                    issue(issues, f"closed loop: {identifier} official metrics missing or hash-changed for {condition}")
                    continue
                info = read_json(info_path) or {}
                per_task = {row.get("task_id"): row for row in info.get("per_task", [])}
                if set(per_task) != expected_tasks or info.get("overall", {}).get("n_episodes") != 50 * len(expected_tasks):
                    issue(issues, f"closed loop: {identifier} official task/episode coverage differs for {condition}")
                    continue
                official_per_task[condition] = per_task
                official_success_granularity[condition] = {}
                for task in expected_tasks:
                    row = per_task[task]
                    raw_successes = [
                        groups.get((task, index), {}).get(condition, {}).get("success")
                        for index in range(50)
                    ]
                    valid, granularity = official_success_row_matches_raw(
                        row,
                        raw_successes,
                        allow_aggregate_only=identifier in ("uva", "uva_joint"),
                    )
                    if not valid:
                        issue(issues, f"closed loop: {identifier} raw episodes disagree with official metrics for {condition} task {task}")
                        break
                    official_success_granularity[condition][str(task)] = granularity
            summary_path = record.get("paired_summary_path")
            if (not summary_path or not Path(summary_path).is_file()
                    or sha256(Path(summary_path)) != record.get("paired_summary_sha256")):
                issue(issues, f"closed loop: {identifier} lacks a pinned paired CI/disagreement summary")
            else:
                summary = read_json(Path(summary_path)) or {}
                successes = {
                    condition: sum(group[condition]["success"] for group in groups.values() if condition in group)
                    for condition in conditions
                }
                if (summary.get("tasks") != len(expected_tasks)
                        or summary.get("initializations_per_task") != 50
                        or summary.get("paired_episodes") != len(episodes)
                        or summary.get("episodes_sha256") != sha256(Path(episodes_path))
                        or summary.get("success_counts") != successes
                        or summary.get("condition_eval_info") != official
                        or (identifier in ("uva", "uva_joint")
                            and summary.get("official_success_granularity")
                                != official_success_granularity)):
                    issue(issues, f"closed loop: {identifier} paired summary does not match raw outcomes")
                pairs = list(itertools.combinations(condition_order, 2))
                disagreement_counts = {}
                action_divergence_counts = {}
                paired_sample_counts = {}
                excluded_initial_observation_mismatch_counts = {}
                expected_ci_keys = set()
                for pair_index, (left, right) in enumerate(pairs):
                    name = f"{left}|{right}"
                    expected_ci_keys.add(name)
                    ordered = sorted(
                        key for key in expected_pairs
                        if (groups.get(key, {}).get(left, {}).get("initial_observation_sha256")
                            == groups.get(key, {}).get(right, {}).get("initial_observation_sha256"))
                    )
                    if any(left not in groups.get(key, {}) or right not in groups.get(key, {}) for key in ordered):
                        continue
                    if not ordered:
                        issue(issues, f"closed loop: {identifier} has no paired resets for {name}")
                        continue
                    paired_sample_counts[name] = len(ordered)
                    excluded_initial_observation_mismatch_counts[name] = len(expected_pairs) - len(ordered)
                    differences = [int(groups[key][right]["success"]) - int(groups[key][left]["success"]) for key in ordered]
                    disagreement_counts[name] = sum(value != 0 for value in differences)
                    action_divergence_counts[name] = sum(
                        groups[key][left]["action_sequence"] != groups[key][right]["action_sequence"]
                        for key in ordered
                    )
                    reported = (summary.get("paired_confidence_intervals") or {}).get(name, {})
                    if (reported.get("method") != "paired_percentile_bootstrap"
                            or reported.get("resamples") != 10_000
                            or reported.get("seed") != 20261005 + pair_index
                            or abs(reported.get("difference", float("inf")) - statistics.fmean(differences)) > 1e-12):
                        issue(issues, f"closed loop: {identifier} paired CI metadata/difference differs for {name}")
                        continue
                    generator = random.Random(20261005 + pair_index)
                    count = len(differences)
                    means = sorted(
                        sum(differences[generator.randrange(count)] for _ in range(count)) / count
                        for _ in range(10_000)
                    )
                    def percentile(fraction):
                        position = fraction * (len(means) - 1)
                        lower = int(position)
                        upper = min(lower + 1, len(means) - 1)
                        return means[lower] + (means[upper] - means[lower]) * (position - lower)
                    expected_interval = (percentile(0.025), percentile(0.975))
                    observed_interval = reported.get("ci95")
                    if (not isinstance(observed_interval, list) or len(observed_interval) != 2
                            or any(not isinstance(value, (int, float)) for value in observed_interval)
                            or any(abs(observed - expected) > 1e-12 for observed, expected in zip(observed_interval, expected_interval))):
                        issue(issues, f"closed loop: {identifier} paired CI differs from raw outcomes for {name}")
                if (set(summary.get("paired_confidence_intervals") or {}) != expected_ci_keys
                        or summary.get("disagreement_counts") != disagreement_counts
                        or summary.get("action_divergence_counts") != action_divergence_counts
                        or summary.get("paired_sample_counts") != paired_sample_counts
                        or summary.get("excluded_initial_observation_mismatch_counts")
                            != excluded_initial_observation_mismatch_counts
                        or sorted(
                            summary.get("cross_implementation_initial_observation_mismatches") or [],
                            key=lambda row: (row.get("task"), row.get("initialization_id")),
                        ) != sorted(
                                cross_implementation_reset_mismatches,
                                key=lambda row: (row["task"], row["initialization_id"]),
                            )):
                    issue(issues, f"closed loop: {identifier} paired disagreements/divergence differ from raw outcomes")
                per_task_summary = summary.get("per_task") or []
                by_task = {
                    row.get("task"): row for row in per_task_summary if isinstance(row, dict)
                }
                if len(per_task_summary) != len(expected_tasks) or set(by_task) != expected_tasks:
                    issue(issues, f"closed loop: {identifier} paired per-task summary is incomplete")
                else:
                    for task in expected_tasks:
                        task_pairs = [(task, initialization) for initialization in range(50)]
                        if any(key not in groups or set(groups[key]) != conditions for key in task_pairs):
                            continue
                        expected_task_success = {
                            condition: sum(groups[key][condition]["success"] for key in task_pairs)
                            for condition in conditions
                        }
                        expected_task_disagreements = {
                            f"{left}|{right}": sum(
                                groups[key][left]["success"] != groups[key][right]["success"]
                                for key in task_pairs
                                if (groups[key][left]["initial_observation_sha256"]
                                    == groups[key][right]["initial_observation_sha256"])
                            )
                            for left, right in pairs
                        }
                        expected_task_paired_counts = {
                            f"{left}|{right}": sum(
                                groups[key][left]["initial_observation_sha256"]
                                == groups[key][right]["initial_observation_sha256"]
                                for key in task_pairs
                            )
                            for left, right in pairs
                        }
                        expected_task_action_divergence = {
                            f"{left}|{right}": sum(
                                groups[key][left]["action_sequence"] != groups[key][right]["action_sequence"]
                                for key in task_pairs
                                if (groups[key][left]["initial_observation_sha256"]
                                    == groups[key][right]["initial_observation_sha256"])
                            )
                            for left, right in pairs
                        }
                        expected_task_ci = {
                            condition: official_per_task[condition][task].get("pc_success_ci95")
                            for condition in conditions if condition in official_per_task
                        }
                        task_row = by_task[task]
                        if (task_row.get("initializations") != 50
                                or task_row.get("success_counts") != expected_task_success
                                or task_row.get("disagreement_counts") != expected_task_disagreements
                                or task_row.get("paired_sample_counts") != expected_task_paired_counts
                                or task_row.get("action_divergence_counts") != expected_task_action_divergence
                                or task_row.get("success_ci95") != expected_task_ci):
                            issue(issues, f"closed loop: {identifier} paired summary differs for task {task}")
    else:
        issue(issues, f"closed loop: unknown evaluator status {status!r} for {identifier}")
    return {"complete": len(issues) == start, "status": status}


def audit_configuration(root: Path, identifier: str) -> dict:
    directory = root / identifier / "full"
    issues: list[str] = []
    capacity = capacity_audit(directory / "diagnostic" / "large_batch_capacity.jsonl", directory, issues)
    components = {
        "replay": replay_audit(directory, issues),
        "large_batch_capacity": capacity,
        "diagnostic": diagnostic_audit(directory, issues, capacity),
        "large_batch_policy_timing": large_policy_timing_audit(directory, capacity, issues),
        "heldout": heldout_audit(directory, issues, capacity),
        "flow_summary": flow_summary_audit(directory / "heldout" / "flow_summary.json", identifier, issues, capacity),
        "operators": operator_audit(directory / "operator" / "raw.jsonl", issues, capacity),
        "ablation": ablation_audit(directory / "diagnostic" / "causal_ablation.jsonl", issues),
        "policy_timing": timing_audit(
            directory / "performance" / "policy_timings.jsonl", "policy", issues,
            required_policy_batches=((1, 2, 4) if identifier == "dexvla" else (1, 2, 4, 8)),
        ),
        "four_mode_timing": timing_audit(directory / "performance" / "serving_timings.jsonl", "serving", issues),
        "large_batch_serving_timing": large_serving_timing_audit(directory, capacity, issues),
        "kernel_launches": launch_audit(directory / "performance" / "kernel_launches.json", issues),
        "serving_queue": queue_audit(
            directory / "serving" / "requests.jsonl", issues,
            logical_batch_size=(2 if identifier == "dexvla" else 4),
            fixed_batch_size=(4 if identifier == "dexvla" else 8),
            serving_timings_path=directory / "performance" / "serving_timings.jsonl",
        ),
        "large_batch_serving_queue": large_serving_queue_audit(directory, capacity, issues),
        "closed_loop": evaluator_audit(directory / "closed_loop" / "evaluator_attempt.json", identifier, issues),
    }
    components["heldout_summary"] = heldout_summary_audit(
        directory / "heldout" / "summary.json", components["heldout"], issues,
    )
    components["serving_queue_summary"] = queue_summary_audit(
        directory / "serving" / "summary.json",
        components["serving_queue"].get("group_counts", {}),
        components["serving_queue"].get("group_metrics", {}), issues,
    )
    if not (directory / "REPORT.md").is_file():
        issue(issues, "report: missing full campaign report")
    return {"id": identifier, "complete": not issues, "issues": issues, "components": components}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    parser.add_argument("--output", type=Path, default=Path("results/cross_model/full_coverage_audit.json"))
    args = parser.parse_args()
    records = [audit_configuration(args.results, identifier) for identifier in CONFIGURATIONS]
    report = {
        "schema_version": 1,
        "authority": "/workspace/batch-invariant-pizero-exp/PLAN.md",
        "configuration_count": len(records),
        "full_complete": sum(record["complete"] for record in records),
        "passed": all(record["complete"] for record in records),
        "configurations": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "configurations"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
