// Small browser-side helpers. Anything that genuinely needs the DOM lives here; everything
// else stays in C# so there is one place to look for behaviour.

window.blazorml = {
    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        try {
            localStorage.setItem('blazorml-theme', theme);
        } catch (e) {
            // Private browsing: the choice just will not persist across visits.
        }
    },

    isDarkTheme() {
        const explicit = document.documentElement.getAttribute('data-theme');
        if (explicit) {
            return explicit === 'dark';
        }
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    },

    async copyText(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (e) {
            // Clipboard access needs a secure context and a user gesture; the caller shows a
            // fallback with the text selectable instead of pretending it worked.
            return false;
        }
    },

    scrollToEnd(element) {
        if (element) {
            element.scrollTop = element.scrollHeight;
        }
    },

    focus(element) {
        if (element) {
            element.focus();
        }
    },

    downloadText(filename, content, mime) {
        const blob = new Blob([content], { type: mime || 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
};
