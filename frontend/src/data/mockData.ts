import type { Chunk, AnalyzeResponse } from '../types';

export const mockChunks: Chunk[] = [
  { id: "c001", startMs: 0,    endMs: 2000, thumbnailUrl: "/thumbs/c001.jpg" },
  { id: "c002", startMs: 2000, endMs: 4000, thumbnailUrl: "/thumbs/c002.jpg" },
  { id: "c003", startMs: 4000, endMs: 6000, thumbnailUrl: "/thumbs/c003.jpg" },
  { id: "c004", startMs: 6000, endMs: 8000, thumbnailUrl: "/thumbs/c004.jpg" },
];

export const mockAnalyzeById: Record<string, AnalyzeResponse> = {
  "c001": {
    clip_id: "demo_clip",
    video_meta: { fps: 30, duration_sec: 2, frames: 60 },
    objects: [
      { name: "person", description: "first person on the left, moving hips and hands" },
      { name: "person", description: "second person on the right, mirroring salsa steps near the first" }
    ],
    relations: [
      ["person", "near", "person"]
    ],
    predicted_actions: [
      { id: "pred1", source: "person", target: "person", action: "will_face_each_other", probability: 0.85 },
      { id: "pred2", source: "person", target: "person", action: "will_extend_hands", probability: 0.72 }
    ],
    narrative: "Two people dance salsa side by side with gentle hip and hand movements."
  },
  "c002": {
    clip_id: "demo_clip",
    video_meta: { fps: 30, duration_sec: 2, frames: 60 },
    objects: [
      { name: "person", description: "two people face each other, extending hands" }
    ],
    relations: [
      ["person", "in_front_of", "person"],
      ["person", "near", "person"]
    ],
    predicted_actions: [
      { id: "pred3", source: "person", target: "person", action: "will_spin", probability: 0.78 },
      { id: "pred4", source: "person", target: "person", action: "will_clap", probability: 0.65 }
    ],
    narrative: "Two people face each other closely, continuing their dance."
  },
  "c003": {
    clip_id: "demo_clip",
    video_meta: { fps: 30, duration_sec: 2, frames: 60 },
    objects: [
      { name: "person", description: "dancer spinning clockwise" },
      { name: "person", description: "partner watching and clapping" },
      { name: "car", description: "blue sedan parked in background" }
    ],
    relations: [
      ["person", "near", "person"],
      ["car", "behind", "person"]
    ],
    predicted_actions: [
      { id: "pred5", source: "person", target: "person", action: "will_bow", probability: 0.92 },
      { id: "pred6", source: "dog", target: "person", action: "will_run_across", probability: 0.45 }
    ],
    narrative: "One dancer performs a spin while their partner watches, with a car visible in the background."
  },
  "c004": {
    clip_id: "demo_clip",
    video_meta: { fps: 30, duration_sec: 2, frames: 60 },
    objects: [
      { name: "person", description: "both dancers bowing to each other" },
      { name: "dog", description: "small dog running across the scene" }
    ],
    relations: [
      ["person", "near", "person"],
      ["dog", "pass", "person"]
    ],
    predicted_actions: [
      { id: "pred7", source: "person", target: "person", action: "will_applaud", probability: 0.88 },
      { id: "pred8", source: "person", target: "person", action: "will_shake_hands", probability: 0.73 }
    ],
    narrative: "The dancers bow to each other as a small dog runs across the scene."
  }
};
