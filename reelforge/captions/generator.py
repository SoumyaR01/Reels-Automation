"""AI-powered caption, hashtag, and engagement optimization engine for social media reels."""

import re
import random
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ReelCaption(BaseModel):
    """Generated social media caption with hashtags and engagement metadata."""
    primary_caption: str
    hook_line: str
    call_to_action: str
    hashtags: List[str] = Field(default_factory=list)
    engagement_score: float = 0.0
    platform_tips: Dict[str, str] = Field(default_factory=dict)
    suggested_posting_times: List[str] = Field(default_factory=list)
    monetization_tags: List[str] = Field(default_factory=list)


class CaptionEngine:
    """Generates optimized captions, hashtags, and engagement metadata for social media reels."""

    # Category-specific hook templates
    HOOK_TEMPLATES = {
        "emotional": [
            "This scene will give you chills 💔",
            "Try watching this without getting emotional 😢",
            "The moment that broke everyone 💫",
            "No one was ready for this scene 🥺",
            "This hit different at 2 AM 🌙",
        ],
        "cinematic": [
            "Cinema at its absolute finest 🎬",
            "This is why we love movies 🔥",
            "Peak filmmaking right here 🎥",
            "The shot that defined the film 📽️",
            "Masterclass in visual storytelling ✨",
        ],
        "action": [
            "The most intense scene of the year 💥",
            "Goosebumps guaranteed 🔥",
            "This sequence is INSANE 🤯",
            "Pure adrenaline in 30 seconds ⚡",
            "Action doesn't get better than this 💪",
        ],
        "inspirational": [
            "Watch this when you need motivation 🏆",
            "The speech that changed everything 💡",
            "This is your sign to keep going 🌟",
            "Sometimes you need to hear this 🙏",
            "The comeback no one expected 🔥",
        ],
        "suspense": [
            "Nobody saw this coming 😱",
            "The twist that left everyone speechless 🕵️",
            "Wait for the ending... 👀",
            "This scene had the whole theater gasping 😨",
            "The tension is UNREAL 🫣",
        ],
        "romantic": [
            "Love at its purest form 💕",
            "The scene that melted hearts 🌹",
            "If this doesn't make you believe in love... 💞",
            "Cinema's most beautiful moment 🥰",
            "This is what peak romance looks like 💫",
        ],
    }

    CTA_TEMPLATES = [
        "Follow for more 🎬",
        "Save this for later! 🔖",
        "Share with someone who needs to see this 💫",
        "Drop a ❤️ if this hit you hard",
        "Tag someone who would love this 🎥",
        "Comment your favorite movie scene below 👇",
        "Turn on notifications for daily reels 🔔",
        "Which scene should I post next? 🤔",
    ]

    # Engagement-optimized hashtag pools
    HASHTAG_POOLS = {
        "universal": [
            "#moviescenes", "#filmtok", "#cinephile", "#movieclips",
            "#moviereels", "#reels", "#viral", "#fyp", "#trending",
            "#movielovers", "#filmcommunity", "#cinema", "#shorts",
        ],
        "emotional": [
            "#emotional", "#heartbreaking", "#cryingscene", "#feeltheemotion",
            "#sadscene", "#moviemoments", "#rightinthefeels", "#tearjerker",
        ],
        "cinematic": [
            "#cinematography", "#filmmaking", "#visualstorytelling",
            "#moviemagic", "#directorcut", "#filmography", "#artofcinema",
        ],
        "action": [
            "#actionscenes", "#actionmovies", "#thrilling", "#epicmoments",
            "#adrenaline", "#stunts", "#fightscene", "#intense",
        ],
        "inspirational": [
            "#motivation", "#inspirational", "#nevergiveup", "#success",
            "#mindset", "#grindset", "#keepgoing", "#hustle", "#believe",
        ],
        "suspense": [
            "#suspense", "#thriller", "#plottwist", "#mystery",
            "#tensionbuilding", "#unexpected", "#mindblown", "#intense",
        ],
        "romantic": [
            "#romance", "#lovestory", "#romanticscene", "#movieromance",
            "#lovequotes", "#relationshipgoals", "#truelove",
        ],
    }

    POSTING_TIMES = {
        "instagram": ["6:00 AM", "12:00 PM", "7:00 PM", "9:00 PM"],
        "youtube_shorts": ["8:00 AM", "12:00 PM", "5:00 PM", "8:00 PM"],
        "tiktok": ["7:00 AM", "10:00 AM", "2:00 PM", "9:00 PM"],
    }

    MONETIZATION_TAGS = [
        "original_content", "transformative_edit",
        "commentary_fair_use", "short_form_vertical",
        "engagement_optimized", "platform_native",
    ]

    # Style to category mapping
    STYLE_MAP = {
        "emotional": ["sad", "romantic"],
        "cinematic": ["powerful_dialogue", "cinematic"],
        "action": ["action"],
        "inspirational": ["motivational"],
        "suspense": ["suspense"],
        "romantic": ["romantic"],
        "sad": ["sad"],
        "powerful_dialogue": ["powerful_dialogue"],
        "motivational": ["motivational"],
    }

    @classmethod
    def map_style_to_category(cls, style: str) -> str:
        """Map a user-friendly style name to the internal category."""
        style_lower = style.lower().strip()
        # Direct matches
        direct = {
            "emotional": "sad",
            "cinematic": "powerful_dialogue",
            "action": "action",
            "inspirational": "motivational",
            "suspense": "suspense",
            "romantic": "romantic",
        }
        return direct.get(style_lower, style_lower)

    @classmethod
    def map_category_to_style(cls, category: str) -> str:
        """Map internal category to user-friendly style name."""
        reverse = {
            "sad": "emotional",
            "powerful_dialogue": "cinematic",
            "action": "action",
            "motivational": "inspirational",
            "suspense": "suspense",
            "romantic": "romantic",
            "cinematic": "cinematic",
        }
        return reverse.get(category.lower(), category)

    @classmethod
    def generate_caption(
        cls,
        style: str,
        title: str = "",
        movie_title: str = "",
        duration: float = 30.0,
        dialogue_excerpt: str = "",
        beat_count: int = 5,
        music_direction: str = "",
    ) -> ReelCaption:
        """Generate an optimized social media caption with hashtags and engagement metadata."""
        style_key = style.lower().strip()
        # Normalize style key for lookup
        for k in cls.HOOK_TEMPLATES:
            if style_key in k or k in style_key:
                style_key = k
                break
        if style_key not in cls.HOOK_TEMPLATES:
            style_key = "cinematic"

        # 1. Hook line
        hook = random.choice(cls.HOOK_TEMPLATES[style_key])

        # 2. Call to action
        cta = random.choice(cls.CTA_TEMPLATES)

        # 3. Build primary caption
        caption_parts = [hook]
        if movie_title:
            caption_parts.append(f"🎬 {movie_title}")
        if dialogue_excerpt:
            short_quote = dialogue_excerpt[:120].strip()
            if short_quote:
                caption_parts.append(f'"{short_quote}"')
        caption_parts.append("")
        caption_parts.append(cta)

        primary = "\n".join(caption_parts)

        # 4. Hashtags - mix universal + style-specific
        universal = random.sample(cls.HASHTAG_POOLS["universal"], min(6, len(cls.HASHTAG_POOLS["universal"])))
        style_tags = cls.HASHTAG_POOLS.get(style_key, [])
        specific = random.sample(style_tags, min(5, len(style_tags))) if style_tags else []

        # Movie title hashtag
        movie_hash = []
        if movie_title:
            clean = re.sub(r'[^a-zA-Z0-9]', '', movie_title.split('.')[0].replace(' ', ''))
            if clean:
                movie_hash = [f"#{clean.lower()}"]

        all_hashtags = list(dict.fromkeys(movie_hash + specific + universal))  # dedupe

        # 5. Engagement score (heuristic based on content richness)
        score = 0.5
        if dialogue_excerpt:
            score += 0.15
        if movie_title:
            score += 0.1
        if beat_count >= 4:
            score += 0.1
        if 25 <= duration <= 35:
            score += 0.1  # Sweet spot for short-form
        score = min(1.0, score + random.uniform(0.0, 0.05))

        # 6. Platform tips
        tips = {
            "instagram": f"Post as Reel with trending audio. Use {len(all_hashtags)} hashtags. Best times: {', '.join(cls.POSTING_TIMES['instagram'][:2])}",
            "youtube_shorts": f"Upload as Short. Add end screen. Description: first 2 lines matter. Best times: {', '.join(cls.POSTING_TIMES['youtube_shorts'][:2])}",
            "tiktok": f"Use native editor for effects. Stitch/Duet enabled. Best times: {', '.join(cls.POSTING_TIMES['tiktok'][:2])}",
        }

        # 7. Posting times
        all_times = []
        for platform, times in cls.POSTING_TIMES.items():
            for t in times[:2]:
                all_times.append(f"{platform}: {t}")

        # 8. Monetization tags
        mon_tags = random.sample(cls.MONETIZATION_TAGS, min(3, len(cls.MONETIZATION_TAGS)))

        return ReelCaption(
            primary_caption=primary,
            hook_line=hook,
            call_to_action=cta,
            hashtags=all_hashtags,
            engagement_score=round(score, 2),
            platform_tips=tips,
            suggested_posting_times=all_times,
            monetization_tags=mon_tags,
        )

    @classmethod
    def get_style_presets(cls) -> List[Dict]:
        """Return available style presets with descriptions."""
        return [
            {
                "id": "emotional",
                "label": "Emotional",
                "icon": "💔",
                "description": "Heartbreak, loss, and deeply moving moments",
                "color": "#38bdf8",
                "categories": ["sad", "romantic"],
            },
            {
                "id": "cinematic",
                "label": "Cinematic",
                "icon": "🎬",
                "description": "Powerful dialogue and visual storytelling",
                "color": "#a855f7",
                "categories": ["powerful_dialogue", "cinematic"],
            },
            {
                "id": "action",
                "label": "Action",
                "icon": "💥",
                "description": "High-octane combat, chases, and intensity",
                "color": "#fb7185",
                "categories": ["action"],
            },
            {
                "id": "inspirational",
                "label": "Inspirational",
                "icon": "🏆",
                "description": "Motivation, triumph, and overcoming failure",
                "color": "#fbbf24",
                "categories": ["motivational"],
            },
            {
                "id": "suspense",
                "label": "Suspense",
                "icon": "🕵️",
                "description": "Mystery, tension, and unexpected twists",
                "color": "#c084fc",
                "categories": ["suspense"],
            },
            {
                "id": "romantic",
                "label": "Romantic",
                "icon": "🌹",
                "description": "Love, tenderness, and heartwarming moments",
                "color": "#f472b6",
                "categories": ["romantic"],
            },
        ]
