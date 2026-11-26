"""Original Hindi Shayari generation engine with strict AI-generated guardrails (Sections 1A, 7, 13)."""

from typing import Dict, List, Optional
from ..planner.schema import HindiShayari


class ShayariEngine:
    """Generates authentic, original Hindi couplets (Sher) in Devanagari script matched to mood and category."""

    # Curated original uncopyrighted Hindi couplets adhering to Urdu/Hindi poetic meter
    CORPUS: Dict[str, List[str]] = {
        "sad": [
            "ख़ामोशियों में भी दर्द की सदा होती है,\nहर टूटे हुए दिल की इक अलग दास्ताँ होती है।",
            "वक़्त की धूप में साए भी छूट जाते हैं,\nअपने ही फैसलों से अक्सर दिल टूट जाते हैं।",
            "कुछ ज़ख़्म ऐसे हैं जो लफ़्ज़ों में बयां नहीं होते,\nआँखों से बहते हैं मगर कभी कम नहीं होते।",
        ],
        "motivational": [
            "गिरकर संभलना ही तो ज़िंदगी का दस्तूर है,\nमंज़िल उसी की है जिसका हौसला बुलंद और पुरनूर है।",
            "अंधेरों से डरकर कभी रास्ते नहीं बदलते,\nतूफ़ानों के आगे कभी इरादे नहीं झुकते।",
            "हार से जो सीखे वही इतिहास रचता है,\nमेहनत के पसीने से ही मुकद्दर चमकता है।",
        ],
        "romantic": [
            "तेरी खामोश निगाहों में एक अजीब सा असर है,\nतुझसे मिलकर ही तो मुकम्मल ये ज़िन्दगी का सफ़र है।",
            "तेरी मुस्कान में छुपी है मेरी सारी खुशियाँ,\nतेरे साथ ही तो रोशन हैं ये महकी हुई रातियाँ।",
        ],
        "action": [
            "तूफ़ानों से टकराने का हौसला रखते हैं हम,\nअंगारों पर चलकर भी नया इतिहास रचते हैं हम।",
            "मैदान-ए-जंग में जब कदम आगे बढ़ाते हैं,\nलहू के हर कतरे से नया जोश जगाते हैं।",
        ],
        "powerful_dialogue": [
            "बातें वही सच्ची हैं जो दिल से निकलती हैं,\nमुसीबतों की आग में ही ज़िंदगियाँ बदलती हैं।",
            "वक़्त हर किसी को आजमाता है इम्तिहान लेकर,\nपहचान वही बनाता है जो लड़े जान देकर।",
        ],
        "suspense": [
            "सायों के पीछे भी कोई राज़ गहरा छुपा है,\nहर मोड़ पर अनजाना सा कोई साया खड़ा है।",
            "सन्नाटे की चादर में कोई साज़िश पल रही है,\nअंधेरे की राहों में कोई आहट चल रही है।",
        ],
    }

    @classmethod
    def generate_shayari(
        cls,
        category: str,
        start_time: float = 22.0,
        end_time: float = 30.0,
    ) -> HindiShayari:
        """Generate mood-matched Hindi shayari with strict AI-attribution metadata."""
        cat_key = category.lower().replace(" ", "_")
        options = cls.CORPUS.get(cat_key, cls.CORPUS["motivational"])
        # Select first or rotate based on hash
        selected_text = options[0]

        return HindiShayari(
            language="hi",
            text=selected_text,
            ai_generated=True,
            placement="overlay_or_voiceover",
            reel_start_seconds=round(start_time, 1),
            reel_end_seconds=round(end_time, 1),
        )


def generate_original_shayari(category: str) -> List[str]:
    """Backward compatibility wrapper returning lines of shayari."""
    obj = ShayariEngine.generate_shayari(category)
    return obj.text.split("\n")

