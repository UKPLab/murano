// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Murano',
			logo: { src: './src/assets/logo.png' },
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/UKPLab/murano' }],
			expressiveCode: {
				themes: ['github-light', 'github-dark'],
				styleOverrides: {
					borderColor: 'transparent',
					borderRadius: '0.375rem',
					frames: {
						shadowColor: 'transparent',
					},
				},
			},
			customCss: [
				'@fontsource/inter/400.css',
				'@fontsource/inter/600.css',
				'@fontsource/inter/800.css',
				'@fontsource/dm-mono/400.css',
				'./src/styles/vitesse.css',
				'./src/styles/dot.css',
			],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Overview', slug: 'getting-started/overview' },
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quickstart', slug: 'getting-started/quickstart' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Recording Activations', slug: 'guides/record' },
						{ label: 'Activation Patching', slug: 'guides/patch' },
						{ label: 'Steering Vectors', slug: 'guides/steer' },
						{ label: 'Probing', slug: 'guides/probe' },
						{ label: 'Pipelines', slug: 'guides/pipeline' },
					],
				},
				{
					label: 'API Reference',
					autogenerate: { directory: 'reference' },
				},
				{
					label: 'Reproduction Gallery',
					items: [
						{ label: 'Overview', slug: 'reproductions' },
					],
				},
			],
		}),
	],
});
