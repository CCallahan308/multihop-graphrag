"""Relations follow entity positions within each sentence."""

import pytest

from src.extraction import RuleBasedExtractor
from src.models import Chunk


@pytest.mark.parametrize(
    ("text", "source", "target", "forward", "reverse", "unrelated"),
    [
        (
            "Company X is a global technology conglomerate established in 1998. "
            "John Doe founded Company X and served as its initial Chief Executive Officer for twenty years.",
            "John Doe", "Company X", "FOUNDED", "FOUNDED_BY", "Chief Executive Officer",
        ),
        (
            "Solaris is a science fiction film. "
            "Andrei Tarkovsky directed Solaris with cinematography by Vadim Yusov.",
            "Andrei Tarkovsky", "Solaris", "DIRECTED", "DIRECTED_BY", "Vadim Yusov",
        ),
    ],
)
def test_relations_use_sentence_order_after_an_earlier_entity_mention(
    text, source, target, forward, reverse, unrelated
):
    result = RuleBasedExtractor().extract(Chunk(chunk_id="sentence-order", text=text))
    relations = {
        (relation.source_entity, relation.target_entity, relation.predicate)
        for relation in result.relationships
    }

    assert (source, target, forward) in relations
    assert (target, source, reverse) in relations
    assert (source, unrelated, forward) not in relations
    assert (unrelated, source, reverse) not in relations
