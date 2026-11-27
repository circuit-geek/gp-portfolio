import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.date(),
    author: z.string().optional(),
    tags: z.array(z.string()).optional(),
    // Set `draft: true` to keep a post out of the build while you are writing it.
    draft: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    image: z.string().optional(),
    tech: z.array(z.string()),
    github: z.string().optional(),
    demo: z.string().optional(),
    arxiv: z.string().optional(),
    track: z.enum(['data-science', 'backend', 'both']).default('data-science'),
    badge: z.string().optional(),
    order: z.number().default(99),
    featured: z.boolean().default(false),
  }),
});

export const collections = { blog, projects };