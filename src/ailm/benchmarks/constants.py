"""Shared constants, seeds and imports for the benchmark package.

Split out so every other module can depend on the same seed and window
definitions without importing one another.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ailm.evaluation import STAGE1_EVALUATION_TEXT
from ailm.loopback_http import validate_loopback_endpoint

DEFAULT_SEED = 20260807

#: v4 differs from v3 in two ways that both change what a number means, which is
#: why it is a new protocol rather than the same one run again: the letter arm is
#: a census of the whole deduplicated pool instead of a 400-position sample, and
#: each probe's calibration now carries a direct two-rule reconstruction check
#: alongside the captured-mass gate. v3 values are retired rather than kept
#: beside these, because a reader comparing them would be comparing sample sizes
#: and procedures at once.
PROTOCOL_VERSION = "external-benchmark-v4"
#: The reliability arm makes two byte-identical server requests.  They must not
#: collapse onto one memoized key, so each trial has an explicit cache-purpose
#: namespace while retaining the same request payload, model identity, and
#: endpoint.  These are cache namespaces, not different server protocols.
#:
#: These two strings are *prefixes*, not finished protocol tags. Both trials are
#: acquired live and then stored, so a second canonical recorder run re-asks the
#: server the same two questions. The premise the probe measures under is matched
#: server state; a model reload or an intervening workload breaks that premise,
#: the second run's reply differs from the stored one, and the store refuses to
#: overwrite an existing key. That refusal is correct and stays exactly as it is.
#: What was wrong was asking it to overwrite at all: the key named the purpose
#: but not the run, so two runs competed for one key. Appending a run scope gives
#: each run its own two keys, and a re-run appends rather than collides.
#:
#: :func:`repeat_trial_protocols` builds one run's tags. The scope is chosen by
#: the recorder that owns the run and passed down explicitly, because a clock
#: read inside the measurement layer would make the cache identity of a call
#: depend on when the function happened to execute.
REPEAT_TRIAL_PROTOCOLS = tuple(f"{PROTOCOL_VERSION}/repeat-trial-{trial}" for trial in (1, 2))
# Resolve once per process so requests, configuration and replay share one origin.
# An explicitly empty or invalid override is an error, never a default request.
OLLAMA_ENDPOINT_ENV = "AILM_OLLAMA_ENDPOINT"
OLLAMA_URL = validate_loopback_endpoint(
    os.environ.get(OLLAMA_ENDPOINT_ENV, "http://localhost:11434")
)
#: Fixed serving choice, carried in every local generation payload and cache key.
#: This avoids the observed local memory-mapped loader stall without changing
#: a model tag, weight identity, prompt, seed or scoring protocol.
OLLAMA_USE_MMAP = False
OLLAMA_NUM_CTX = 8192
OLLAMA_PROMPT_BYTE_TOKEN_FACTOR = 3
OLLAMA_TEMPLATE_TOKEN_RESERVE = 4096
OLLAMA_CONTEXT_BUDGET_POLICY = "utf8-byte-upper-bound-plus-template-reserve-v1"


def repeat_trial_protocols(run_scope: str) -> tuple[str, ...]:
    """Return the two repeat-trial cache namespaces belonging to one run.

    The scope is checked structurally only — non-empty, unpadded, and free of
    the ``/`` that separates a namespace from its scope — so that the tags stay
    parseable back into (prefix, scope). Whether a particular scope is one a
    *publication* run may claim is a recorder policy, enforced where the receipt
    configuration is validated rather than here.
    """

    if not isinstance(run_scope, str) or not run_scope or run_scope != run_scope.strip():
        raise ValueError("repeat-trial run scope must be a non-empty unpadded string")
    if "/" in run_scope:
        raise ValueError("repeat-trial run scope must not contain a namespace separator")
    return tuple(f"{prefix}/{run_scope}" for prefix in REPEAT_TRIAL_PROTOCOLS)


#: Sampled positions per task, bounded above by the deduplicated letter pool and
#: reachable downward through ``--letter-n``/``--phoneme-n``/``--g2p-n``.
#:
#: This number is a scientific choice, not a runtime knob, and lowering it to
#: shorten a test run silently changes what the benchmark can conclude. At 100
#: positions the phoneme and G2P contrasts stop clearing the corrected family
#: threshold entirely; at 400 every one of them clears it by more than five
#: orders of magnitude. The threshold itself is not repeated here — it is
#: ``bench_family_alpha``, which the recorder derives from the registered model
#: count, so a literal copied into this comment would go stale the next time a
#: model is added or removed. A sample chosen for wall-clock convenience costs
#: the paper its findings while leaving every test green.
#:
#: The letter arm is underpowered at any affordable sample — roughly 0.02 power
#: at the corrected threshold here, and ~n=6000 (about 11 hours per run) to
#: reach 0.80 — and the supplement says so in those words rather than implying
#: the sample settles it. 400 is therefore the point where the resolvable arms
#: resolve and the unresolvable one is honestly labeled.
#:
#: Consequence for ``test_rerunning_the_recorders_is_a_no_op``: this size costs
#: roughly a quarter of an hour of live model calls per registered model, so
#: that test's one-hour budget covers this recorder and little else. Raise its
#: timeout or scope it away from the Ollama-backed recorders; do not buy it back
#: by shrinking the measurement.
DEFAULT_TASK_N = 400

#: Sentinel for the letter arm: score every distinct position in the fixture
#: rather than a sample of them.
#:
#: The letter arm is the only one whose contrast was ever underpowered. At the
#: 400 positions the phoneme and G2P arms need to be decisive (p <= 1e-09), the
#: letter contrast reaches 0.024 power against the corrected threshold, so a
#: letter null said nothing. Raising just this arm to the whole deduplicated
#: pool takes it to 0.967 (decision 30's power table), and turns it from a
#: sample into a census: there is no subsampling left to argue about, and the
#: seed stops mattering for this arm because nothing is being chosen.
#:
#: This is deliberately a sentinel rather than the literal pool size. Writing
#: 7952 here would be a second copy of a property of the fixture, free to drift
#: from it the first time the passage changes, and the drift would look like a
#: sample rather than an error.
LETTER_CENSUS = -1

#: Prefix windows for the letter context sweep. The comparator is order-1, so
#: only the probe can use the extra context — that asymmetry is the experiment.
LETTER_WINDOW_SHORT = 40
LETTER_WINDOW_LONG = 160

#: Calibration sample and binning. Ollama caps ``top_logprobs`` at 20, below the
#: 30-symbol letter alphabet, so the probe distribution is completed rather than
#: observed in full (see :func:`_distribution_from_candidates`).
DEFAULT_CALIBRATION_N = 200
CALIBRATION_BINS = 10
TOP_LOGPROBS = 20

#: Minimum share of the answer-slot distribution that must be *observed* rather
#: than assumed before calibration is reportable for a model.
#:
#: RETIRED FROM GATING (decision 93): the measurability decision is now the
#: direct dual-rule divergence check (:func:`dual_rule_calibration`) rather
#: than this floor, which the offline sweep measured as a non-conservative
#: proxy for rule-dependence. The constant is retained as provenance — the
#: starved-fraction diagnostic and the floor-locating sweep both cite it — but
#: it no longer decides whether a calibration figure is reported.
#:
#: The reported candidates are the raw next-token distribution truncated at
#: ``top_logprobs``, so a model whose mass sits mostly on tokens the response
#: grammar forbids yields a distribution that is mostly completion. Completion
#: pushes the distribution toward its floor, which lowers mean confidence, and
#: on a low-accuracy model a lowered confidence *reduces* the calibration error:
#: a reconstruction failure would be reported as good calibration. The threshold
#: is one half because that is the point at which the reported distribution
#: stops being mostly a measurement, not because of where any model happened to
#: land — a model below it is not badly calibrated, it is not measured.
#:
#: That argument is the reason for the value; it is not evidence for it, and the
#: project's own offline sweep (:func:`completion_rule_sweep`, recorded under the
#: ``bench_floor_`` labels) found the argument wrong in the direction that
#: matters. Captured mass is only a proxy for the quantity the floor is meant to
#: control — how much of a reported calibration number the completion rule rather
#: than the model decides — and on AILM's own letter columns the two completion
#: rules still disagree by more than the declared negligibility threshold well
#: above one half. The floor is left at one half deliberately rather than raised
#: to the located value: the located floor needs more candidates than this
#: endpoint will return, so raising the gate here would withhold every model's
#: calibration rather than measure it better. What replaces the proxy is the
#: direct per-model check (:func:`dual_rule_calibration`), which is recorded for
#: every probe whether or not this floor admits it.
CALIBRATION_MASS_FLOOR = 0.5

#: Target captured masses for the offline experiment that locates the floor
#: above empirically (:func:`completion_rule_sweep`). The three measured probes
#: sit at roughly 0.07, 0.86 and 1.00 captured mass, so the live protocol
#: measures nothing at all between about 12% and 98% and the floor's value is
#: argued rather than observed. The sweep spans that gap on distributions that
#: need no model call: AILM's own letter transition columns, which are real
#: next-symbol distributions over the same thirty-symbol inventory the benchmark
#: scores. The grid starts below the lowest measured system and ends at total
#: capture, where truncation is the identity and the two completion rules must
#: agree exactly.
CALIBRATION_FLOOR_GRID_LOW = 0.05
CALIBRATION_FLOOR_GRID_HIGH = 1.0
CALIBRATION_FLOOR_GRID_STEP = 0.01

#: Declared negligibility threshold for the disagreement between two completion
#: rules, in the units of the metrics the benchmark reports. One hundredth is
#: the second decimal place the calibration table is read at, so a disagreement
#: below it cannot move a published figure at the precision it is published.
#: Declared here, next to the grid, rather than chosen after the curve was seen:
#: a threshold picked from the plot locates whichever floor was wanted.
CALIBRATION_FLOOR_EPSILON = 0.01

#: Protocol tag for the offline floor experiment. Versioned apart from
#: ``PROTOCOL_VERSION`` because it makes no model call and shares nothing with
#: the live benchmark except the letter inventory and the scoring rules.
CALIBRATION_FLOOR_PROTOCOL = "calibration-floor-v1"

#: How far apart the two completion rules may be at total capture, where
#: truncation removes nothing and they must reconstruct the same distribution.
#: Not zero, and the reason is worth stating: :func:`complete_min_bound` routes
#: through the candidate payload the server would have sent, so every
#: probability makes a ``log`` then ``exp`` round trip that the observed-only
#: rule does not. The residual is a few units in the last place. Measuring the
#: shipped rule is worth more than bit-exactness against a rule that skips it,
#: and a tolerance this tight cannot hide a real disagreement — the smallest one
#: anywhere else on the grid is fourteen orders of magnitude larger.
CALIBRATION_FLOOR_IDENTITY_TOLERANCE = 1e-12

#: Seed-ensemble sizes. Both replicate seeds run at temperature 0, where the
#: expected spread is exactly zero; the point is to measure that, not assume it.
DEFAULT_ENSEMBLE_N = 120
DEFAULT_REPEAT_N = 30
SEED_OFFSETS = (1, 2)

#: Alpha for the retrospective power diagnostic. The benchmark's paired p-values
#: are documented exclusions from the repository significance family
#: (``tests/test_statistics.py``), so applying that family's Bonferroni-corrected
#: threshold here would report power against a correction this comparison is not
#: subject to. The conventional two-sided 0.05 is used and recorded as a claim.
POWER_ALPHA = 0.05

#: Gold-phone-count buckets for the G2P length stratification, as
#: ``(name, low, high)`` with an inclusive, exhaustive, non-overlapping cover.
G2P_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("short", 0, 5),
    ("mid", 6, 9),
    ("long", 10, 1_000_000),
)

# The benchmark's name for the Stage 1 evaluation passage. The fixture itself has
# one owner, :mod:`ailm.evaluation`, where its bytes, source, and license are
# authenticated; this binding exists so the benchmark modules and the web export
# name the passage by its role here without copying it.
HELD_OUT_TEXT = STAGE1_EVALUATION_TEXT


#: A substantially larger, non-repeating public-domain held-out passage for
#: the external letter benchmark. The Stage 1 gate fixture (above) is a short
#: passage repeated three times, which caps the letter pool at a few hundred
#: distinct positions — too few to resolve a reference-sized letter contrast at
#: the corrected family alpha (see Lane C's power analysis). This passage is
#: longer by an order of magnitude and deliberately non-repeating, so the
#: distinct-position pool grows with the text and the long context window stays
#: below the passage period. All lines are public-domain folklore (classic
#: tongue twisters and nursery rhymes, pre-1928).
HELD_OUT_TEXT_LARGE = (
    "betty botter bought some butter but she said the butter-s bitter if i put it in my batter it will make my batter bitter "
    "swan swam over the sea swim swan swim swan swam back again well swum swan "
    "i scream you scream we all scream for ice cream "
    "six slippery snails slid slowly seaward "
    "the thirty-three thieves thought that they thrilled the throne throughout thursday "
    "one-one was a race horse two-two was one too one-one won one race two-two won one too "
    "six thick thistle sticks six thick thistles stick "
    "a proper copper coffee pot "
    "crisp crusts crackle and crunch "
    "fresh fried fish fish fresh fried fried fish fresh fish fried "
    "red lorry yellow lorry red lorry yellow lorry "
    "lesser leather never weathered wetter weather better "
    "truly rural truly rural truly rural "
    "irish wristwatch irish wristwatch irish wristwatch "
    "toy boat toy boat toy boat toy boat "
    "six sick hicks nick six slick bricks with picks and sticks "
    "pad kid poured curd pulled cod "
    "brisk brave brigadiers brandished broad bright blades blunderbusses and bludgeons balancing them badly "
    "can you can a can as a canner can can a can "
    "top cop sipping sloppy slops "
    "amid the mists and coldest frosts with barest wrists and stoutest boasts he thrusts his fists against the posts and still insists he sees the ghosts "
    "which wristwatches are swiss wristwatches "
    "chop shops stock chops "
    "pete-s pa pete said to pete yes it-s tricky "
    "a flea and a fly flew up in a flue said the flea let us fly said the fly let us flee "
    "denise sees the fleece at least denise could sneeze and feed and freeze the fleas "
    "theo the thistle sifter sifted a sieve of unsifted thistles "
    "erie the earwig ate an ear of corn "
    "moses supposes his toeses are roses but moses supposes erroneously "
    "esau wood sawed wood all the wood esau wood saw esau sought to saw "
    "a tutor who tooted the flute tried to tutor two tooters to toot said the two to the tutor is it harder to toot or to tutor two tooters to toot "
    "the skunk sat on a stump and thunk the stump stunk but the stump thunk the skunk stunk "
    "nine nice night nurses nursing nicely "
    "pearls please pretty petunia "
    "gray geese in the green grass grazing "
    "rory the warrior and roger the worrier were reared wrongly in a rural brewery "
    "rhys watched ross switch irish wristwatches for swiss wristwatches "
    "wally wilson washed his woolen walrus with a wooden washer "
    "mildred mears made a mess of her mother-s meal "
    "humphrey humpbacked horace hunchbacked hugh humped his hump home "
    "a big black bug bit a big black bear and made the big black bear bleed blood "
    "the sixth sheik-s sixth sheep-s sick "
    "don-t trouble trouble until trouble troubles you "
    "a tree toad loved a she-toad who lived up in a tree "
    "two toads totally tired of trying to trot to tetsbury "
    "larry hurley a burly squirrel hurler hurled a furry squirrel "
    "eddie edited it eddie edited it eddie edited it "
    "kate and duane drove a dune buggy "
    "zizi zestfully zipped through the zany zoo zone "
    "amy and may made a mint of meringue on monday "
    "bobby bippy bought a bat bobby bippy bought a ball "
    "willie wisps whistle while they work "
    "gus the gassy goose gobbled the gooey gum "
    "hazel hazed the hazy henhouse hazily "
    "ivy-s icy igloo is in an icy isle "
    "jim-jam jelly in a jam jar jiggles gently "
    "queen quinn quickly quizzed the quiet quail "
    "rex the ram ran round the rugged rocks "
    "sue sang softly in the sunny shallows "
    "tess tossed the tassels to the tepid tanners "
    "victor vastly vexed the vast verdant valley "
    "wendy waved a wavy wand at the wandering wombat "
    "xavier examined the x-ray of the exotic ox "
    "yvonne yawned at the yawning yak "
    "zack zealously zipped the zebra-s zipper "
    "one two buckle my shoe three four shut the door "
    "jack and jill went up the hill to fetch a pail of water jack fell down and broke his crown and jill came tumbling after "
    "hey diddle diddle the cat and the fiddle the cow jumped over the moon the little dog laughed to see such sport and the dish ran away with the spoon "
    "little miss muffet sat on a tuffet eating her curds and whey along came a spider who sat down beside her and frightened miss muffet away "
    "twinkle twinkle little star how i wonder what you are up above the world so high like a diamond in the sky "
    "baa baa black sheep have you any wool yes sir yes sir three bags full "
    "old mother hubbard went to the cupboard to get her poor dog a bone but when she got there the cupboard was bare so the poor dog had none "
    "little bo peep has lost her sheep and can-t tell where to find them leave them alone and they-ll come home wagging their tails behind them "
    "jack be nimble jack be quick jack jump over the candlestick "
    "georgie porgie pudding and pie kissed the girls and made them cry when the boys came out to play georgie porgie ran away "
    "hickory dickory dock the mouse ran up the clock the clock struck one the mouse ran down hickory dickory dock "
    "hot cross buns hot cross buns one a penny two a penny hot cross buns "
    "peter peter pumpkin eater had a wife and couldn-t keep her he put her in a pumpkin shell and there he kept her very well "
    "polly put the kettle on polly put the kettle on we-ll all have tea "
    "ride a cock horse to banbury cross to see a fine lady upon a white horse "
    "sing a song of sixpence a pocket full of rye four and twenty blackbirds baked in a pie "
    "three blind mice three blind mice see how they run see how they run they all ran after the farmer-s wife who cut off their tails with a carving knife did you ever see such a sight in your life as three blind mice "
    "simple simon met a pieman going to the fair said simple simon to the pieman let me taste your ware "
    "little boy blue come blow your horn the sheep-s in the meadow the cow-s in the corn "
    "ding dong bell pussy-s in the well who put her in little johnny green who pulled her out little tommy stout "
    "pat-a-cake pat-a-cake baker-s man bake me a cake as fast as you can "
    "rock-a-bye baby on the tree top when the wind blows the cradle will rock "
    "wee willie winkie runs through the town upstairs and downstairs in his nightgown "
    "old king cole was a merry old soul and a merry old soul was he he called for his pipe and he called for his bowl and he called for his fiddlers three "
    "mary mary quite contrary how does your garden grow with silver bells and cockle shells and pretty maids all in a row "
    "doctor foster went to gloucester in a shower of rain he stepped in a puddle right up to his middle and never went there again "
    "little jack horner sat in the corner eating a christmas pie he put in his thumb and pulled out a plum and said what a good boy am i "
    "rain rain go away come again another day little johnny wants to play "
    "star light star bright first star i see tonight i wish i may i wish i might have the wish i wish tonight "
    "there was an old woman who lived in a shoe she had so many children she didn-t know what to do "
    "tom tom the piper-s son stole a pig and away did run the pig was eat and tom was beat and tom went crying down the street "
    "tweedledum and tweedledee agreed to have a battle for tweedledum said tweedledee had spoiled his nice new rattle "
    "yankee doodle went to town riding on a pony he stuck a feather in his cap and called it macaroni "
    "i had a little nut tree nothing would it bear but a silver nutmeg and a golden pear "
    "little tommy tucker sings for his supper what shall he eat white bread and butter "
    "goosie goosie gander whither shall i wander upstairs and downstairs and in my lady-s chamber "
    "jack spratt could eat no fat his wife could eat no lean and so between them both you see they licked the platter clean "
    "the man in the moon came tumbling down and asked his way to norwich he went by the south and burned his mouth with sucking cold pease porridge "
    "little polly flinders sat among the cinders warming her pretty little toes her mother came and caught her and whipped her little daughter for spoiling her nice new clothes "
    "pease porridge hot pease porridge cold pease porridge in the pot nine days old "
    "robin the bobbin the big-bellied ben he ate more meat than fourscore men "
    "the lion and the unicorn were fighting for the crown the lion beat the unicorn all round about the town "
)


#: The passage the external letter benchmark scores. The Stage 1 gate fixture is
#: kept for the gate's own protocol; the benchmark uses the larger non-repeating
#: passage so its letter contrasts have enough distinct positions to resolve.
LETTER_TEXT = HELD_OUT_TEXT_LARGE


@dataclass(frozen=True)
class ProbeSpec:
    """Identity of one served external reference probe.

    ``key`` names ledger suffixes, ``tag`` is the mutable Ollama selector used
    for calls, and ``display_name`` is reader-facing text.  Keeping all three in
    one registry prevents a recorder, figure, and manuscript from silently
    describing different systems under the same suffix.
    """

    key: str
    tag: str
    display_name: str


def validate_served_probes(probes: tuple[ProbeSpec, ...]) -> tuple[ProbeSpec, ...]:
    """Validate the single registry shared by recorders, prose, and figures.

    Empty fields would create unaddressable ledger labels or blank reader-facing
    captions.  Duplicate keys, mutable tags, or display names would collapse two
    served systems onto one namespace, one cache identity, or one legend entry.
    Refuse all of those at import time, before a network call or claim write.
    """

    if not probes:
        raise ValueError("SERVED_PROBES must register at least one probe")

    normalized: dict[str, list[str]] = {
        "key": [],
        "tag": [],
        "display_name": [],
    }
    for index, probe in enumerate(probes):
        if not isinstance(probe, ProbeSpec):
            raise TypeError(f"SERVED_PROBES[{index}] must be ProbeSpec, got {type(probe).__name__}")
        for field in normalized:
            value = getattr(probe, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SERVED_PROBES[{index}].{field} must be nonempty")
            if value != value.strip():
                raise ValueError(
                    f"SERVED_PROBES[{index}].{field} must not contain surrounding whitespace"
                )
            normalized[field].append(value.casefold() if field == "display_name" else value)

    for field, values in normalized.items():
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"SERVED_PROBES {field} values must be unique: {duplicates}")
    return probes


SERVED_PROBES: tuple[ProbeSpec, ...] = validate_served_probes(
    (
        ProbeSpec("gemma3", "gemma3:4b", "Gemma 3 4B"),
        ProbeSpec("qwen25", "qwen2.5:3b", "Qwen 2.5 3B"),
        # TODO Medium: third probe family -- a genuinely non-instruct BASE model,
        # different family from the two registered instruct chat models above.
        # Verified via `ollama show smollm:1.7b-base-v0.2-q4_K_M`: capability is
        # completion-only and the template is the raw Prompt template, so no chat
        # markup mediates any arm's query. Its 2048-token context window bounds
        # every arm's prompt; the synthetic protocols declare an 8192-token
        # budget, but the recorded maximum prompt is ~107 tokens, so the bound is
        # not reached in practice.
        ProbeSpec("smollm", "smollm:1.7b-base-v0.2-q4_K_M", "SmolLM 1.7B Base v0.2"),
    )
)
PROBE_MODEL_TAGS: dict[str, str] = {probe.key: probe.tag for probe in SERVED_PROBES}
PROBE_DISPLAY_NAMES: dict[str, str] = {probe.key: probe.display_name for probe in SERVED_PROBES}
SERVED_PROBE_COUNT = len(SERVED_PROBES)
BASE_MODEL_PROBE_KEYS: frozenset[str] = frozenset({"smollm"})
#: Probes queried with their raw completion template rather than a chat
#: template. The supplement's inventory sentence derives its instruct/base
#: split from this set against SERVED_PROBES, so registering a new probe
#: forces the classification to be made here rather than typed into prose.


HOSTED_SYNTHETIC_MODEL = "meta-llama/llama-3.3-70b-instruct"
HOSTED_SYNTHETIC_DISPLAY = "Llama 3.3 70B Instruct"


def hosted_cohort_label(key: str) -> str:
    """Name a retained protocol run key without attributing local model weights."""
    return f"{key} run"
