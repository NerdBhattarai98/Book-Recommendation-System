// Main JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initFavorites();
    initToRead();
    initSearch();
    updateFavoritesCount();
    initNavHandlers();
});

// Favorites
let favorites = [];

function initFavorites() {
    // Only fetch favorites if user is authenticated
    if (!window.IS_AUTHENTICATED) {
        favorites = [];
        return;
    }
    
    fetch('/user/favorites')
        .then(res => res.json())
        .then(data => {
            favorites = data.favorites || [];
            updateFavoritesUI();
            updateFavoritesCount();
        })
        .catch(() => { favorites = []; });
}

function toggleFavorite(isbn) {
    // If user isn't authenticated, redirect to login (preserve next)
    if (!window.IS_AUTHENTICATED || window.IS_AUTHENTICATED === 'false') {
        const next = window.location.pathname + window.location.search;
        window.location.href = '/auth/login?next=' + encodeURIComponent(next);
        return;
    }

    const isFavorite = favorites.includes(isbn);
    const endpoint = isFavorite ? '/user/favorites/remove' : '/user/favorites/add';

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ isbn: isbn })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            favorites = data.favorites;
            updateFavoritesUI();
            updateFavoritesCount();
            showToast(isFavorite ? 'Removed from favorites' : 'Added to favorites');
        }
    })
    .catch(err => console.error('Error:', err));
}

function initNavHandlers() {
    const favBtn = document.getElementById('favorites-nav-btn');
    if (favBtn) {
        favBtn.addEventListener('click', function(e) {
            // navigate to favorites page; server will redirect to login if needed
            window.location.href = '/favorites';
        });
    }
}

// To-Read List
let toRead = [];

function initToRead() {
    // Only fetch to-read list if user is authenticated
    if (!window.IS_AUTHENTICATED) {
        toRead = [];
        return;
    }
    
    fetch('/user/to_read')
        .then(res => res.json())
        .then(data => {
            toRead = data.to_read || [];
            updateToReadUI();
        })
        .catch(() => { toRead = []; });
}

function toggleToRead(isbn) {
    if (!window.IS_AUTHENTICATED) {
        const next = window.location.pathname + window.location.search;
        window.location.href = '/auth/login?next=' + encodeURIComponent(next);
        return;
    }

    const isInList = toRead.includes(isbn);
    const endpoint = isInList ? '/user/to_read/remove' : '/user/to_read/add';

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ isbn: isbn })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            toRead = data.to_read;
            updateToReadUI();
            showToast(isInList ? 'Removed from To Read list' : 'Added to To Read list');
        }
    })
    .catch(err => console.error('Error:', err));
}

function updateToReadUI() {
    document.querySelectorAll('.to-read-btn, .to-read-btn-large').forEach(btn => {
        const isbn = btn.dataset.isbn;
        const text = btn.querySelector('.btn-text');
        if (toRead.includes(isbn)) {
            btn.classList.add('active');
            if (text) text.textContent = 'Remove from To Read';
        } else {
            btn.classList.remove('active');
            if (text) text.textContent = 'Add to To Read';
        }
    });
}

function updateFavoritesUI() {
    document.querySelectorAll('.favorite-btn, .favorite-btn-large').forEach(btn => {
        const isbn = btn.dataset.isbn;
        const text = btn.querySelector('.btn-text');
        
        if (favorites.includes(isbn)) {
            btn.classList.add('active');
            if (text) text.textContent = 'Remove from Favorites';
        } else {
            btn.classList.remove('active');
            if (text) text.textContent = 'Add to Favorites';
        }
    });
}

function updateFavoritesCount() {
    const countEl = document.getElementById('nav-favorites-count');
    if (countEl) countEl.textContent = favorites.length;
}

function showToast(message) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed; bottom: 24px; right: 24px;
        background: #1e293b; color: white;
        padding: 16px 24px; border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000; animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

// Add toast animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    @keyframes slideOut { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }
`;
document.head.appendChild(style);

// Search
function initSearch() {
    setupAutocomplete('search-input', 'search-suggestions');
    setupAutocomplete('hero-search-input', 'hero-suggestions');
}

function setupAutocomplete(inputId, suggestionsId) {
    const input = document.getElementById(inputId);
    const suggestions = document.getElementById(suggestionsId);
    
    if (!input || !suggestions) return;
    
    let timeout;
    
    input.addEventListener('input', function() {
        clearTimeout(timeout);
        const query = this.value.trim();
        
        if (query.length < 2) {
            suggestions.classList.remove('active');
            return;
        }
        
        timeout = setTimeout(() => {
            fetch(`/search?q=${encodeURIComponent(query)}`, {
                headers: { 'Accept': 'application/json' }
            })
            .then(res => res.json())
            .then(data => {
                if (data.length > 0) {
                    suggestions.innerHTML = data.slice(0, 6).map(book => `
                        <div class="suggestion-item" data-isbn="${escapeHtml(book.isbn)}">
                            <div class="suggestion-title">${escapeHtml(book.title)}</div>
                            <div class="suggestion-author">${escapeHtml(book.author)}</div>
                        </div>
                    `).join('');
                    suggestions.classList.add('active');
                    
                    suggestions.querySelectorAll('.suggestion-item').forEach(item => {
                        item.addEventListener('click', function() {
                            window.location.href = `/book/${this.dataset.isbn}`;
                        });
                    });
                } else {
                    suggestions.classList.remove('active');
                }
            })
            .catch(() => suggestions.classList.remove('active'));
        }, 300);
    });
    
    document.addEventListener('click', function(e) {
        if (!input.contains(e.target) && !suggestions.contains(e.target)) {
            suggestions.classList.remove('active');
        }
    });
    
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') suggestions.classList.remove('active');
    });
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
