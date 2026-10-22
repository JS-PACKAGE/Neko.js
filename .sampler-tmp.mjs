import { LogitsSampler, Tensor } from '@huggingface/transformers';
const V = 248320;
const sampler = LogitsSampler.getSampler({ do_sample: true, top_k: 50 });
const greedy = LogitsSampler.getSampler({ do_sample: false });
const data = new Float32Array(V).map(() => Math.random() * 20 - 10);
const logits = new Tensor('float32', data, [1, V]);
for (const [name, s] of [['multinomial', sampler], ['greedy', greedy]]) {
  for (let i = 0; i < 5; i++) await s.sample(logits);
  const t = performance.now(); for (let i = 0; i < 50; i++) await s.sample(logits); console.log(name, ((performance.now() - t) / 50).toFixed(2), 'ms');
}
