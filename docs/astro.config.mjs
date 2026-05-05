// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	// Default GitHub Pages URL for UKPLab/murano. Switch `site` to a custom
	// domain (e.g. https://murano.ukp.informatik.tu-darmstadt.de) and set
	// `base: '/'` once DNS + repo Pages settings are configured.
	site: 'https://ukplab.github.io',
	base: '/murano',
	integrations: [
		starlight({
			components: {
				ThemeSelect: './src/components/ThemeSelect.astro',
				SocialIcons: './src/components/SiteNav.astro',
			},
			title: 'Murano',
			logo: { light: './src/assets/logo-light.png', dark: './src/assets/logo-dark.png' },
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
						{ label: 'Overview', slug: 'docs/getting-started/overview' },
						{ label: 'Installation', slug: 'docs/getting-started/installation' },
						{ label: 'Quickstart', slug: 'docs/getting-started/quickstart' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Recording Activations', slug: 'docs/guides/record' },
						{ label: 'Activation Patching', slug: 'docs/guides/patch' },
						{ label: 'Steering Vectors', slug: 'docs/guides/steer' },
						{ label: 'Probing', slug: 'docs/guides/probe' },
						{ label: 'Pipelines', slug: 'docs/guides/pipeline' },
					],
				},
				{
					label: 'Reproductions',
					items: [
						{ label: 'Overview', slug: 'docs/reproductions/overview' },
						{ label: 'Submit a Reproduction', slug: 'docs/reproductions/submit' },
					],
				},
				{
					label: 'API Reference',
					autogenerate: { directory: 'docs/reference' },
				},
			],
		}),
	],
});
