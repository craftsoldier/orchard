//! Canonical action pairing, as the draft ZIP "Canonical Action Pairing" specifies.
//!
//! A canonical build derives the dummy notes, the `rseed` of each output without a
//! caller-supplied one, and the order of the actions from a secret pairing key, an
//! application context and the nullifiers of the real spends. The same inputs then give
//! the same nullifier, note commitment, `epk` and note ciphertext in every action.

use alloc::vec::Vec;

use blake2b_simd::{Hash, Params};
use ff::PrimeField;
use rand::Rng;

use super::{BuildError, OutputInfo, SpendInfo};
use crate::{
    keys::{FullViewingKey, Scope, SpendingKey},
    note::{Note, NoteVersion, RandomSeed, RandomSeedBytes, Rho},
    spec::to_base,
    tree::MerklePath,
    value::NoteValue,
};

const SEED_PERSONALIZATION: &[u8; 16] = b"Zcash_PairingSd_";
const PRF_PERSONALIZATION: &[u8; 16] = b"Zcash_PairingPRF";
const ORDER_PERSONALIZATION: &[u8; 16] = b"Zcash_PairingOrd";

const TAG_DUMMY_SPEND_KEY: u8 = 0x01;
const TAG_DUMMY_SPEND_RHO: u8 = 0x02;
const TAG_DUMMY_SPEND_RSEED: u8 = 0x03;
const TAG_OUTPUT_RSEED: u8 = 0x04;
const TAG_DUMMY_OUTPUT_KEY: u8 = 0x05;
const TAG_DUMMY_OUTPUT_RSEED: u8 = 0x06;

const ORDER_SPEND: u8 = 0x00;
const ORDER_OUTPUT: u8 = 0x01;

/// The ZIP 302 "no memo" encoding that every dummy output carries.
const NO_MEMO: [u8; 512] = {
    let mut memo = [0; 512];
    memo[0] = 0xF6;
    memo
};

/// The inputs of a canonical build that the caller supplies (see
/// [`Builder::set_canonical_pairing`](super::Builder::set_canonical_pairing)).
#[derive(Clone, Debug)]
pub(super) struct CanonicalPairing {
    key: [u8; 32],
    ctx: Vec<u8>,
}

/// Spends and outputs, each with its index in the [`BundleMetadata`](super::BundleMetadata)
/// numbering, in action order.
type Paired = (Vec<(usize, SpendInfo)>, Vec<(usize, OutputInfo)>);

impl CanonicalPairing {
    pub(super) fn new(key: [u8; 32], ctx: &[u8]) -> Self {
        CanonicalPairing {
            key,
            ctx: ctx.to_vec(),
        }
    }

    /// Pads `spends` and `outputs` to `num_actions` and orders both, as the ZIP specifies.
    ///
    /// `outputs` are in the caller's order: the plain outputs, then the change outputs.
    /// When a derived value is invalid, the build repeats with the retry counter appended
    /// to the context, as section "Failed values" requires.
    pub(super) fn pair(
        &self,
        note_version: NoteVersion,
        num_actions: usize,
        spends: Vec<SpendInfo>,
        outputs: Vec<OutputInfo>,
        mut rng: impl Rng,
    ) -> Result<Paired, BuildError> {
        let mut real_nullifiers: Vec<[u8; 32]> = spends
            .iter()
            .map(|spend| spend.note.nullifier(&spend.fvk).to_bytes())
            .collect();
        let spend_nullifiers = real_nullifiers.clone();
        real_nullifiers.sort_unstable();

        for retry in 0..=u8::MAX {
            let mut ctx = self.ctx.clone();
            if retry > 0 {
                ctx.push(retry);
            }
            let seed = Seed::derive(&self.key, &ctx, &real_nullifiers);
            if let Some(order) = seed.order(
                note_version,
                num_actions,
                &spend_nullifiers,
                &outputs,
                &mut rng,
            )? {
                return Ok(order.assemble(spends, outputs));
            }
        }
        // 256 consecutive failures, each with probability about 2^-245, cannot happen.
        Err(BuildError::InvalidRandomSeed)
    }
}

/// The pairing seed $d$ of one attempt.
struct Seed([u8; 32]);

/// The result of one successful attempt, before the real spends and outputs move into it.
struct Order {
    dummy_spends: Vec<SpendInfo>,
    output_rseeds: Vec<RandomSeedBytes>,
    dummy_outputs: Vec<OutputInfo>,
    /// Indices into real spends followed by dummy spends, in action order.
    spend_order: Vec<usize>,
    /// Indices into real outputs followed by dummy outputs, in action order.
    output_order: Vec<usize>,
}

impl Seed {
    fn derive(key: &[u8; 32], ctx: &[u8], sorted_nullifiers: &[[u8; 32]]) -> Self {
        let mut h = Params::new()
            .hash_length(32)
            .personal(SEED_PERSONALIZATION)
            .to_state();
        h.update(key);
        h.update(&(ctx.len() as u64).to_le_bytes());
        h.update(ctx);
        h.update(&(sorted_nullifiers.len() as u32).to_le_bytes());
        for nf in sorted_nullifiers {
            h.update(nf);
        }
        Seed(left_32(&h.finalize()))
    }

    /// $\mathsf{PRF^{pair}}_d(t, i, c)$.
    fn prf(&self, tag: u8, index: u32, counter: u32) -> Hash {
        Params::new()
            .hash_length(64)
            .personal(PRF_PERSONALIZATION)
            .to_state()
            .update(&self.0)
            .update(&[tag])
            .update(&index.to_le_bytes())
            .update(&counter.to_le_bytes())
            .finalize()
    }

    fn prf_32(&self, tag: u8, index: u32) -> [u8; 32] {
        left_32(&self.prf(tag, index, 0))
    }

    /// $\mathsf{PairingKey}_d(t, i)$ and the address at diversifier index 0 of its
    /// external full viewing key.
    fn pairing_key(&self, tag: u8, index: u32) -> (SpendingKey, FullViewingKey) {
        let sk = (0..=u32::MAX)
            .find_map(|c| SpendingKey::from_bytes(left_32(&self.prf(tag, index, c))).into())
            .expect("a valid spending key occurs with probability about 1 - 2^-250");
        let fvk = FullViewingKey::from(&sk);
        (sk, fvk)
    }

    /// Dummy spend `i`, or `None` if its note is invalid.
    fn dummy_spend(
        &self,
        index: u32,
        note_version: NoteVersion,
        rng: &mut impl Rng,
    ) -> Option<SpendInfo> {
        let (sk, fvk) = self.pairing_key(TAG_DUMMY_SPEND_KEY, index);
        let recipient = fvk.address_at(0u32, Scope::External);
        let rho_bytes: [u8; 64] = *self.prf(TAG_DUMMY_SPEND_RHO, index, 0).as_array();
        let rho = Rho::from_bytes(&to_base(&rho_bytes).to_repr())
            .expect("ToBase gives a canonical field element");
        let rseed = RandomSeed::from_bytes(self.prf_32(TAG_DUMMY_SPEND_RSEED, index), &rho)
            .into_option()?;
        let note =
            Note::from_parts(recipient, NoteValue::ZERO, rho, rseed, note_version).into_option()?;
        Some(SpendInfo {
            dummy_sk: Some(sk),
            fvk,
            scope: Scope::External,
            note,
            // The Action statement checks the path only for a note of nonzero value.
            merkle_path: Some(MerklePath::dummy(rng)),
        })
    }

    fn dummy_output(&self, index: u32, note_version: NoteVersion) -> OutputInfo {
        let (_, fvk) = self.pairing_key(TAG_DUMMY_OUTPUT_KEY, index);
        OutputInfo::new_with_rseed(
            None,
            fvk.address_at(0u32, Scope::External),
            NoteValue::ZERO,
            note_version,
            NO_MEMO,
            RandomSeedBytes::from_bytes(self.prf_32(TAG_DUMMY_OUTPUT_RSEED, index)),
        )
    }

    fn order_key(&self, kind: u8, value: &[u8; 32]) -> [u8; 32] {
        let hash = Params::new()
            .hash_length(32)
            .personal(ORDER_PERSONALIZATION)
            .to_state()
            .update(&self.0)
            .update(&[kind])
            .update(value)
            .finalize();
        left_32(&hash)
    }

    /// Derives every value of one attempt. Returns `Ok(None)` when a derived value is
    /// invalid, and an error when two sort keys collide.
    fn order(
        &self,
        note_version: NoteVersion,
        num_actions: usize,
        spend_nullifiers: &[[u8; 32]],
        outputs: &[OutputInfo],
        rng: &mut impl Rng,
    ) -> Result<Option<Order>, BuildError> {
        let mut dummy_spends = Vec::with_capacity(num_actions - spend_nullifiers.len());
        for i in 0..num_actions - spend_nullifiers.len() {
            match self.dummy_spend(index(i), note_version, rng) {
                Some(spend) => dummy_spends.push(spend),
                None => return Ok(None),
            }
        }
        let nullifiers: Vec<[u8; 32]> = spend_nullifiers
            .iter()
            .copied()
            .chain(
                dummy_spends
                    .iter()
                    .map(|spend| spend.note.nullifier(&spend.fvk).to_bytes()),
            )
            .collect();

        let output_rseeds: Vec<RandomSeedBytes> = outputs
            .iter()
            .enumerate()
            .map(|(j, output)| {
                output.rseed.unwrap_or_else(|| {
                    RandomSeedBytes::from_bytes(self.prf_32(TAG_OUTPUT_RSEED, index(j)))
                })
            })
            .collect();
        let dummy_outputs: Vec<OutputInfo> = (0..num_actions - outputs.len())
            .map(|i| self.dummy_output(index(i), note_version))
            .collect();

        let spend_order = self.sort(ORDER_SPEND, nullifiers.iter())?;
        let output_order = self.sort(
            ORDER_OUTPUT,
            output_rseeds
                .iter()
                .map(RandomSeedBytes::as_bytes)
                .chain(dummy_outputs.iter().map(|output| {
                    output
                        .rseed
                        .as_ref()
                        .expect("dummy outputs carry an rseed")
                        .as_bytes()
                })),
        )?;

        // Each output takes ρ from the nullifier of the spend it pairs with.
        for (&s, &o) in spend_order.iter().zip(&output_order) {
            let rho = Rho::from_bytes(&nullifiers[s]).expect("a nullifier is a field element");
            let (output, rseed) = match o.checked_sub(outputs.len()) {
                None => (&outputs[o], output_rseeds[o]),
                Some(i) => (
                    &dummy_outputs[i],
                    dummy_outputs[i]
                        .rseed
                        .expect("dummy outputs carry an rseed"),
                ),
            };
            let valid = RandomSeed::from_bytes(*rseed.as_bytes(), &rho)
                .into_option()
                .and_then(|rseed| {
                    Note::from_parts(output.recipient, output.value, rho, rseed, note_version)
                        .into_option()
                })
                .is_some();
            if !valid {
                return Ok(None);
            }
        }

        Ok(Some(Order {
            dummy_spends,
            output_rseeds,
            dummy_outputs,
            spend_order,
            output_order,
        }))
    }

    /// Indices of `values` in ascending order of their sort keys.
    fn sort<'a>(
        &self,
        kind: u8,
        values: impl Iterator<Item = &'a [u8; 32]>,
    ) -> Result<Vec<usize>, BuildError> {
        let mut keyed: Vec<([u8; 32], usize)> = values
            .enumerate()
            .map(|(i, value)| (self.order_key(kind, value), i))
            .collect();
        keyed.sort_unstable();
        if keyed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(BuildError::DuplicatePairingKey);
        }
        Ok(keyed.into_iter().map(|(_, i)| i).collect())
    }
}

impl Order {
    fn assemble(self, spends: Vec<SpendInfo>, mut outputs: Vec<OutputInfo>) -> Paired {
        for (output, rseed) in outputs.iter_mut().zip(self.output_rseeds) {
            output.rseed = Some(rseed);
        }
        let mut spends: Vec<Option<SpendInfo>> = spends
            .into_iter()
            .chain(self.dummy_spends)
            .map(Some)
            .collect();
        let mut outputs: Vec<Option<OutputInfo>> = outputs
            .into_iter()
            .chain(self.dummy_outputs)
            .map(Some)
            .collect();
        (
            take(&mut spends, self.spend_order),
            take(&mut outputs, self.output_order),
        )
    }
}

/// The items at `order`, each with its index.
fn take<T>(items: &mut [Option<T>], order: Vec<usize>) -> Vec<(usize, T)> {
    order
        .into_iter()
        .map(|i| (i, items[i].take().expect("each index occurs once")))
        .collect()
}

fn left_32(hash: &Hash) -> [u8; 32] {
    hash.as_bytes()[..32]
        .try_into()
        .expect("the hash has at least 32 bytes")
}

fn index(i: usize) -> u32 {
    u32::try_from(i).expect("a bundle has fewer than 2^32 actions")
}

#[cfg(all(test, feature = "circuit"))]
mod tests {
    use alloc::vec::Vec;

    use rand::{rand_core::UnwrapErr, rngs::StdRng, rngs::SysRng, SeedableRng};

    use super::{Seed, NO_MEMO, ORDER_OUTPUT, ORDER_SPEND, TAG_OUTPUT_RSEED};
    use crate::{
        builder::{BuildError, Builder, BundleType},
        bundle::{Authorized, Bundle, BundleVersion, Flags},
        keys::{FullViewingKey, Scope, SpendAuthorizingKey, SpendingKey},
        note::{Note, Nullifier, RandomSeedBytes, Rho},
        tree::MerklePath,
        value::NoteValue,
        Address, Anchor,
    };

    const KEY: [u8; 32] = [7; 32];
    const VALUE: u64 = 10_000;

    struct Inputs {
        sk: SpendingKey,
        fvk: FullViewingKey,
        recipient: Address,
        note: Note,
        path: MerklePath,
        anchor: Anchor,
    }

    /// One spendable note from a fixed seed.
    fn inputs(seed: u8) -> Inputs {
        let mut rng = StdRng::from_seed([seed; 32]);
        let sk = SpendingKey::from_bytes([seed; 32]).unwrap();
        let fvk = FullViewingKey::from(&sk);
        let recipient = fvk.address_at(0u32, Scope::External);
        let version = BundleVersion::ironwood_v3();
        let rho = Rho::from_nf_old(Nullifier::dummy(&mut rng));
        let note = Note::new(
            recipient,
            NoteValue::from_raw(VALUE),
            rho,
            version.note_version(),
            &mut rng,
        );
        let path = MerklePath::dummy(&mut rng);
        let anchor = path.root(note.commitment().into());
        Inputs {
            sk,
            fvk,
            recipient,
            note,
            path,
            anchor,
        }
    }

    fn bound_rseed(n: u8) -> RandomSeedBytes {
        RandomSeedBytes::from_bytes([n; 32])
    }

    /// One real spend, two outputs with a provided `rseed`, one plain output and one change
    /// output: four actions, three of them with a dummy spend.
    fn builder(inputs: &Inputs, pairing: Option<(&[u8; 32], &[u8])>) -> Builder {
        let version = BundleVersion::ironwood_v3();
        let mut builder = Builder::new(
            BundleType::DEFAULT,
            version,
            version.default_flags(),
            inputs.anchor,
        )
        .unwrap();
        builder
            .add_spend(inputs.fvk.clone(), inputs.note, inputs.path.clone())
            .unwrap();
        for n in 1..=2 {
            builder
                .add_output_with_rseed(
                    None,
                    inputs.recipient,
                    NoteValue::ZERO,
                    [n; 512],
                    bound_rseed(n),
                )
                .unwrap();
        }
        builder
            .add_output(None, inputs.recipient, NoteValue::from_raw(1_000), [3; 512])
            .unwrap();
        builder
            .add_change_output(
                inputs.fvk.clone(),
                None,
                inputs.recipient,
                NoteValue::from_raw(VALUE - 1_000),
                [4; 512],
            )
            .unwrap();
        if let Some((key, ctx)) = pairing {
            builder.set_canonical_pairing(*key, ctx);
        }
        builder
    }

    /// (nullifier, cmx, epk, note ciphertext) of every action, in action order.
    type Fingerprint = Vec<([u8; 32], [u8; 32], [u8; 32], Vec<u8>)>;

    fn fingerprint(builder: Builder, build_seed: u8) -> Fingerprint {
        let (bundle, _) = builder
            .build::<i64>(StdRng::from_seed([build_seed; 32]))
            .unwrap()
            .unwrap();
        bundle
            .actions()
            .iter()
            .map(|a| {
                (
                    a.nullifier().to_bytes(),
                    a.cmx().to_bytes(),
                    a.encrypted_note().epk_bytes,
                    a.encrypted_note().enc_ciphertext.as_ref().to_vec(),
                )
            })
            .collect()
    }

    fn canonical(inputs: &Inputs, key: &[u8; 32], ctx: &[u8], build_seed: u8) -> Fingerprint {
        fingerprint(builder(inputs, Some((key, ctx))), build_seed)
    }

    #[test]
    fn canonical_build_reproduces_every_action() {
        let inputs = inputs(1);
        let first = canonical(&inputs, &KEY, b"epoch 1", 1);
        assert_eq!(first.len(), 4);
        assert_eq!(canonical(&inputs, &KEY, b"epoch 1", 2), first);

        // The standard build samples dummies and order at random.
        assert_ne!(
            fingerprint(builder(&inputs, None), 1),
            fingerprint(builder(&inputs, None), 2)
        );
    }

    #[test]
    fn canonical_build_depends_on_key_context_and_spends() {
        let dummy_nullifiers = |fp: &Fingerprint, real: &[u8; 32]| {
            fp.iter()
                .map(|(nf, _, _, _)| *nf)
                .filter(|nf| nf != real)
                .collect::<Vec<_>>()
        };
        let one = inputs(1);
        let nf = one.note.nullifier(&one.fvk).to_bytes();
        let base = dummy_nullifiers(&canonical(&one, &KEY, b"epoch 1", 1), &nf);
        assert_eq!(base.len(), 3);
        assert_ne!(
            dummy_nullifiers(&canonical(&one, &[8; 32], b"epoch 1", 1), &nf),
            base
        );
        assert_ne!(
            dummy_nullifiers(&canonical(&one, &KEY, b"epoch 2", 1), &nf),
            base
        );

        // Another real spend under the same key and context gives other dummy spends, so
        // two bundles that can both be mined never share a dummy nullifier.
        let two = inputs(2);
        let nf_two = two.note.nullifier(&two.fvk).to_bytes();
        let other = dummy_nullifiers(&canonical(&two, &KEY, b"epoch 1", 1), &nf_two);
        assert!(other.iter().all(|nf| !base.contains(nf)));
    }

    #[test]
    fn canonical_build_follows_the_zip() {
        let inputs = inputs(1);
        let nf = inputs.note.nullifier(&inputs.fvk).to_bytes();
        let seed = Seed::derive(&KEY, b"epoch 1", &[nf]);
        let (bundle, meta) = builder(&inputs, Some((&KEY, b"epoch 1")))
            .build_for_pczt(UnwrapErr(SysRng))
            .unwrap();
        let actions = &bundle.actions;

        // Spends and outputs are in ascending order of their keyed sort keys.
        let spend_keys: Vec<_> = actions
            .iter()
            .map(|a| seed.order_key(ORDER_SPEND, &a.spend.nullifier.to_bytes()))
            .collect();
        assert!(spend_keys.is_sorted());
        let output_keys: Vec<_> = actions
            .iter()
            .map(|a| seed.order_key(ORDER_OUTPUT, a.output.rseed.unwrap().as_bytes()))
            .collect();
        assert!(output_keys.is_sorted());

        // Provided seeds are kept; the plain output (index 2) and the change output
        // (index 3) take PRF(0x04, j).
        for (j, expected) in [
            *bound_rseed(1).as_bytes(),
            *bound_rseed(2).as_bytes(),
            seed.prf_32(TAG_OUTPUT_RSEED, 2),
            seed.prf_32(TAG_OUTPUT_RSEED, 3),
        ]
        .iter()
        .enumerate()
        {
            let action = &actions[meta.output_action_index(j).unwrap()];
            assert_eq!(action.output.rseed.unwrap().as_bytes(), expected);
            assert!(action.output.verify_note_commitment(&action.spend).is_ok());
        }

        // Dummy spends are the ZIP's notes: the pairing key's address and value 0.
        for i in 0..3 {
            let expected = seed.dummy_spend(i, inputs.note.version(), &mut UnwrapErr(SysRng));
            let expected = expected.unwrap();
            let nf = expected.note.nullifier(&expected.fvk);
            assert!(actions.iter().any(|a| a.spend.nullifier == nf));
        }
    }

    #[test]
    fn dummy_outputs_carry_no_memo() {
        let seed = Seed([3; 32]);
        let output = seed.dummy_output(0, BundleVersion::ironwood_v3().note_version());
        assert_eq!(output.memo, NO_MEMO);
        assert_eq!(output.value, NoteValue::ZERO);
        assert!(output.ovk.is_none());
    }

    #[test]
    fn canonical_pairing_refuses_cross_address_disabled_bundles() {
        let inputs = inputs(1);
        let version = BundleVersion::ironwood_v3();
        let flags = Flags::from_parts(true, true, false);
        let mut builder = Builder::new(BundleType::DEFAULT, version, flags, inputs.anchor).unwrap();
        builder
            .add_spend(inputs.fvk.clone(), inputs.note, inputs.path)
            .unwrap();
        builder.set_canonical_pairing(KEY, b"");
        assert!(matches!(
            builder.build::<i64>(UnwrapErr(SysRng)),
            Err(BuildError::CrossAddressDisabled)
        ));
    }

    #[test]
    fn canonical_pairing_refuses_duplicate_rseeds() {
        let inputs = inputs(1);
        let mut builder = builder(&inputs, Some((&KEY, b"")));
        builder
            .add_output_with_rseed(
                None,
                inputs.recipient,
                NoteValue::ZERO,
                [0; 512],
                bound_rseed(1),
            )
            .unwrap();
        assert!(matches!(
            builder.build::<i64>(UnwrapErr(SysRng)),
            Err(BuildError::DuplicatePairingKey)
        ));
    }

    #[test]
    fn canonical_bundle_proves_and_verifies() {
        let version = BundleVersion::ironwood_v3();
        let keys = crate::cached_test_keys(version.circuit_version());
        let mut rng = UnwrapErr(SysRng);
        let sighash = [0; 32];
        let inputs = inputs(1);

        let (unauthorized, _) = builder(&inputs, Some((&KEY, b"epoch 1")))
            .build::<i64>(&mut rng)
            .unwrap()
            .unwrap();
        let bundle: Bundle<Authorized, i64> = unauthorized
            .create_proof(keys.proving_key(), &mut rng)
            .unwrap()
            .apply_signatures(rng, sighash, &[SpendAuthorizingKey::from(&inputs.sk)])
            .unwrap();
        assert!(bundle.verify_proof(keys.verifying_key()).is_ok());
        for action in bundle.actions() {
            assert_eq!(action.rk().verify(&sighash, action.authorization()), Ok(()));
        }

        // Every output decrypts to a note whose rseed rebuilds its commitment.
        let ivk = inputs.fvk.to_ivk(Scope::External);
        let decrypted = (0..bundle.actions().len())
            .filter_map(|i| bundle.decrypt_output_with_key(i, &ivk))
            .count();
        assert_eq!(decrypted, 4);
    }
}
