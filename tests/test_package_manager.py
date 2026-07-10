import unittest

from utils.torch_install_helper import NvidiaGpuInfo, prepare_torch_install_request


class PackageManagerTest(unittest.TestCase):
    def test_torch_install_request_rewrites_plain_torch_for_selected_cuda(self):
        request = prepare_torch_install_request(
            ["torch", "torchvision", "einops"],
            gpu_detector=lambda: [NvidiaGpuInfo("RTX 4090", 8.9)],
            torch_device="cuda",
            torch_cuda_version="cu128",
        )

        self.assertEqual(request.device, "cuda")
        self.assertEqual(request.cuda_version, "cu128")
        self.assertIn("torch==2.10.0", request.requirements)
        self.assertIn("torchvision==0.25.0", request.requirements)
        self.assertIn("einops", request.requirements)
        self.assertEqual(request.env["INDEX_URL"], "https://download.pytorch.org/whl/cu128")


if __name__ == "__main__":
    unittest.main()
