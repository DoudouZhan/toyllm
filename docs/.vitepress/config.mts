import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/toyllm/',
  title: "ToyLLM项目文档",
  description: "ToyLLM项目文档",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      // { text: 'Examples', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: 'ToyLLM项目文档导航',
        items: [
          { text: '项目使用文档', link: '/usage' },
          { text: '大模型相关知识学习记录', link: '/preliminary' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/henryzhuhr/toyllm' }
    ],
    externalLinkIcon: true,
    footer: {
      message: '基于 <a href="https://choosealicense.com/licenses/gpl-3.0/">GPL-3.0</a> 许可发布',
      copyright: `版权所有 © 2024-${new Date().getFullYear()} <a href="https://github.com/HenryZhuHR?tab=repositories">HenryZhuHR</a>`
    },
    outline: {
      label: '页面导航'
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },
    langMenuLabel: '多语言',
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '文档导航',
    darkModeSwitchLabel: '深色模式开关',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  },
  lastUpdated: true,
  markdown: {
    math: true
  }
})
