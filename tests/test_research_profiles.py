import unittest

from astock_core.strategies.research_profiles import get_research_profile, research_profile_catalog


class ResearchProfileTests(unittest.TestCase):
    def test_catalog_exposes_short_and_long_choices_without_internal_checks(self):
        catalog = research_profile_catalog()

        self.assertEqual({item["周期标签"] for item in catalog}, {"短线", "中长线", "长线"})
        self.assertTrue(all("验证项" not in item for item in catalog))
        self.assertTrue(all("外部 Skill" not in item for item in catalog))
        self.assertTrue(all("来源说明" not in item for item in catalog))
        self.assertTrue(all("适用边界" not in item for item in catalog))

    def test_all_thirty_master_profiles_are_registered_with_period_and_source(self):
        masters = [
            item for item in research_profile_catalog(include_details=True)
            if item.get("外部 Skill", "").endswith("hgsz2003/master30）")
        ]

        self.assertEqual(len(masters), 30)
        self.assertTrue(all(item["周期标签"] in {"中长线", "长线"} for item in masters))
        self.assertTrue(any("巴菲特" in item["名称"] for item in masters))

    def test_unknown_profile_safely_uses_default(self):
        self.assertEqual(get_research_profile("unknown")["id"], "short_term_momentum")


if __name__ == "__main__":
    unittest.main()
