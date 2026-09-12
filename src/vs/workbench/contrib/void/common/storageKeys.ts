/*--------------------------------------------------------------------------------------
 *  Copyright 2026 The A-Tech Corporation PTY LTD. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

// past values:
// 'void.settingsServiceStorage'
// 'void.settingsServiceStorageI' // 1.0.2

// 1.0.3
export const VOID_SETTINGS_STORAGE_KEY = 'void.settingsServiceStorageII'


// past values:
// 'void.chatThreadStorage'
// 'void.chatThreadStorageI' // 1.0.2
// 'void.chatThreadStorageII' // 1.0.3 - added size-based limits

// Current: 1.0.4 - added storage versioning and migration support
export const THREAD_STORAGE_KEY = 'void.chatThreadStorageIII'

// Per-thread storage (1.0.5): each thread is stored under its own key
// (`void.chatThread.<threadId>`) so a save only serializes the thread that
// changed instead of re-stringify-ing every stored thread. The legacy blob key
// above is kept as a migration source only.
export const THREAD_STORAGE_KEY_PREFIX = 'void.chatThread.'

// Storage version key - used for migration tracking
export const THREAD_STORAGE_VERSION_KEY = 'void.chatThreadStorage.version'
export const CURRENT_THREAD_STORAGE_VERSION = 1

// Learning progress storage key
export const LEARNING_PROGRESS_STORAGE_KEY = 'void.learningProgressStorage'



export const OPT_OUT_KEY = 'void.app.optOutAll'

// User identification & cohorts - Analytics enhancement
export const USER_EMAIL_KEY = 'void.app.userEmail';
export const USER_ID_KEY = 'void.app.userId';
export const COHORT_KEY = 'void.app.cohort';
export const FIRST_SESSION_KEY = 'void.app.firstSession';
export const LAST_SESSION_KEY = 'void.app.lastSession';

// Session tracking - Analytics enhancement
export const SESSION_ID_KEY = 'void.app.sessionId';
export const SESSION_START_TIME_KEY = 'void.app.sessionStartTime';

// Multi-workspace agent manager
export const WORKSPACE_REGISTRY_STORAGE_KEY = 'void.workspaceRegistryStorage';
export const WORKSPACE_HUB_PORT_KEY = 'void.workspaceHubPort';

// A-Coder OAuth config
export const ACODER_AUTH_STORAGE_KEY = 'void.aCoderAuthStorage';

// What's New modal - tracks last seen version
export const WHATS_NEW_LAST_VERSION_KEY = 'void.whatsNew.lastVersion';

// Implementation plans - persisted across restarts (per-thread plans)
export const IMPLEMENT_PLANS_STORAGE_KEY = 'void.implementationPlansStorage';