# Work on JIRA Ticket - Engineer

## Atlassian MCP Configuration

You are a principal engineer with equity stake in the company building this software.

You're working on $ARGUMENTS that you can find information about in JIRA using the atlassian MCP.

This workflow uses the Atlassian MCP integration with the following project-specific configuration:

**Organization**: yqm0nk3y1.atlassian.net
**Cloud ID**: `5c7e1404-fb71-4e23-b0e7-5a45d3c7db8c`
**Project Key**: `ADF` (AnomalyDetectionFramework project)
**Issue Types**: Epic, Task

### Common JIRA Status Transitions

The AnomalyDetectionFramework project uses these status transitions:

- **To Do** → **In Progress** (start work)
- **In Progress** → **Review** (PR created)
- **Review** → **Done** (PR merged)


Note: Always fetch available transitions first as IDs may vary.

## Pre-flight Checklist

Before starting any work, verify:

1. **Environment Setup**
   - Atlassian MCP is configured and authenticated
   - GitHub CLI is authenticated (`gh auth status`)
   - All development dependencies are installed
   - Node.js version matches project requirements

2. **Workspace State**
   - Main worktree is clean (`git status`)
   - You're in the correct repository
   - No uncommitted changes in main worktree that could be lost

3. **Access Verification**
   - Can access JIRA ticket
   - Have write access to GitHub repository
   - Development environment is functional

## Initial Setup and Context Loading

### 1. Load Project Context

```bash
# Read and execute prime.md commands
cat .claude/commands/prime.md
# Execute all commands from prime.md
```

### 2. Prepare Main Worktree

```bash
# Navigate to main worktree
cd /path/to/main-project

# Stash any current changes in main worktree
git stash save "WIP: Before starting $ARGUMENTS"

# Ensure main worktree is on main branch
git checkout main

# Pull latest changes
git pull origin main

# Verify clean state
git status
```

### 3. Fetch and Analyze JIRA Ticket

**Fetch comprehensive ticket information:**

- Full ticket details including all fields
- Description with acceptance criteria
- All comments for additional context
- Attachments (designs, documents, etc.)
- Parent epic
- Linked issues (blocks, is blocked by, relates to)


### 4. Update JIRA Status and Comment

```
1. Get available transitions for the ticket
2. Transition to "In Progress"
3. Add comment: "Starting work on this ticket. Estimated completion: [date]"
```

## Ticket Analysis and Planning

### Dependency Analysis

1. **Identify Related Work**
   - Search related issues in the same epic
   - Check for similar recently completed work
   - Review any documented patterns or decisions

## Development Workflow

### Branch Creation and Worktree Setup

```bash
# Create a dedicated worktree for this ticket
# This allows parallel work without disrupting main worktree
git worktree add ../project-$ARGUMENTS feature/$ARGUMENTS-short-description

# Navigate to the new worktree
cd ../project-$ARGUMENTS

# Verify we're on the correct branch
git branch --show-current

# Example structure:
# /path/to/main-project/           <- Main worktree on main branch
# /path/to/project-ADF-123/        <- Feature worktree for ADF-123
# /path/to/project-ADF-124/        <- Another feature worktree for ADF-124 (if needed)
```

### Test-Driven Development Process

#### 1. Test Planning and Design

**Identify test categories needed:**

- **Unit Tests**: Pure functions, business logic, utilities
- **Integration Tests**: API endpoints, database operations, external services
- **Component Tests**: React components in isolation
- **E2E Tests**: Critical user journeys, happy paths

**Test scenario mapping:**

```javascript
// Example test planning structure
describe('Feature: User Authentication', () => {
  describe('Unit Tests', () => {
    // Business logic validation
    // Token generation/validation
    // Permission checking
  });
  
  describe('Integration Tests', () => {
    // API endpoint behavior
    // Database interactions
    // External service mocking
  });
  
  describe('Component Tests', () => {
    // Login form behavior
    // Error state handling
    // Loading states
  });
  
  describe('E2E Tests', () => {
    // Complete login flow
    // Session persistence
    // Logout functionality
  });
});
```

#### 2. Write Comprehensive Tests First

**Test file naming convention:**
- Unit/Integration: `[filename].test.ts`
- Components: `[ComponentName].test.tsx`
- E2E: `[feature].e2e.test.ts`

**Test structure requirements:**

```typescript
// Clear test descriptions
it('should authenticate user with valid credentials', async () => {
  // Arrange: Set up test data
  const validUser = { email: 'test@example.com', password: 'secure123' };
  
  // Act: Perform the action
  const result = await authenticateUser(validUser);
  
  // Assert: Verify expectations
  expect(result.success).toBe(true);
  expect(result.token).toBeDefined();
});

// Edge cases and error scenarios
it('should reject authentication with invalid password', async () => {
  // Test implementation
});

// Performance considerations
it('should complete authentication within 200ms', async () => {
  // Test implementation
});
```

#### 3. Implement Minimal Solution

**Implementation guidelines:**

1. **Write just enough code to pass tests**
2. **Follow existing patterns:**
   ```typescript
   // Check existing code for patterns
   // Use established naming conventions
   // Maintain consistent file structure
   ```

3. **Use monorepo packages:**
   ```typescript
   import { SharedUtil } from '@company/shared';
   import { UIComponent } from '@company/ui-kit';
   ```

4. **Ensure cross-platform compatibility:**
   ```typescript
   // Use platform-agnostic code
   // Test Platform.OS specific behavior
   // Document any platform differences
   ```

#### 4. Refactor and Optimize

**Refactoring checklist:**

- [ ] Extract reusable functions to utilities
- [ ] Apply SOLID principles
- [ ] Remove code duplication
- [ ] Improve naming clarity
- [ ] Add proper TypeScript types
- [ ] Optimize performance bottlenecks
- [ ] Add comprehensive error handling

### Code Quality and Standards

#### 1. Automated Quality Checks

```bash
# Run all quality checks in feature worktree
npm run lint        # ESLint checks
npm run type-check  # TypeScript validation
npm run test        # All tests
npm run test:coverage # Coverage report

# Fix auto-fixable issues
npm run lint:fix
```

**Required standards:**
- Zero ESLint errors (warnings documented if unavoidable)
- 100% TypeScript type coverage
- Minimum 80% test coverage for new code
- All existing tests must still pass

#### 2. Security and Child Safety Review

**Critical security checklist:**

- [ ] No PII exposed in logs or error messages
- [ ] Content filtering applied to user-generated content
- [ ] Input validation on all user inputs
- [ ] API rate limiting implemented
- [ ] Authentication/authorization properly checked
- [ ] No hardcoded credentials or secrets

**Code review for safety:**

```typescript
// Bad: Exposing user data
console.log(`User ${user.email} logged in`);

// Good: Safe logging
console.log(`User ${user.id} logged in`);

// Bad: No content filtering
const userMessage = req.body.message;

// Good: Content filtered
const userMessage = await contentFilter.clean(req.body.message);
```

#### 3. Performance Optimization

**Mobile performance checklist:**

- [ ] Minimize React re-renders (use memo, useMemo, useCallback)
- [ ] Implement lazy loading for heavy components
- [ ] Optimize images and assets
- [ ] Use proper list virtualization for long lists
- [ ] Monitor bundle size impact
- [ ] Profile performance on low-end devices

**API optimization:**

```typescript
// Implement caching strategies
const cacheKey = `user_${userId}`;
const cached = await cache.get(cacheKey);
if (cached) return cached;

// Batch API calls when possible
const results = await Promise.all([
  fetchUserData(userId),
  fetchUserPreferences(userId),
  fetchUserStats(userId)
]);

// Monitor API costs
trackAPIUsage('openai', { tokens: response.usage.total_tokens });
```

### Documentation and Commits

#### 1. Atomic Commit Strategy

```bash
# Each commit should be focused and complete
git add src/auth/login.ts src/auth/login.test.ts
git commit -m "feat(auth): implement user login functionality"

git add src/auth/logout.ts src/auth/logout.test.ts  
git commit -m "feat(auth): add logout with session cleanup"

# Use conventional commits
# feat: new feature
# fix: bug fix
# docs: documentation only
# refactor: code change that neither fixes a bug nor adds a feature
# test: adding missing tests
# chore: changes to build process or auxiliary tools
```

#### 2. Documentation Requirements

**Code documentation:**

```typescript
/**
 * Authenticates a user with the provided credentials
 * @param credentials - User email and password
 * @returns Promise resolving to auth result with token or error
 * @throws {AuthenticationError} When credentials are invalid
 * @example
 * const result = await authenticateUser({
 *   email: 'user@example.com',
 *   password: 'securePassword123'
 * });
 */
export async function authenticateUser(
  credentials: UserCredentials
): Promise<AuthResult> {
  // Implementation
}
```

**Update relevant documentation:**
- [ ] API documentation for new endpoints
- [ ] README if setup/configuration changes
- [ ] Migration guide for breaking changes
- [ ] Architecture decision records (ADRs) for significant decisions

## Deployment Process

### Pre-deployment Checklist

1. **Code Complete**
   - [ ] All acceptance criteria met
   - [ ] All tests passing
   - [ ] Code quality checks passed
   - [ ] Documentation updated

2. **Testing Complete**
   - [ ] All testing run on bash command passed

### Push and Create Pull Request

#### 1. Push Feature Branch

```bash
# Ensure all changes committed in feature worktree
git status

# Push feature branch to remote
git push origin feature/$ARGUMENTS-description
```

#### 2. Create Comprehensive Pull Request

```bash
gh pr create --title "feat: Description [$ARGUMENTS]" --body "$(cat <<'EOF'
## Summary
Brief description of what this PR accomplishes and why it's needed.

## Changes
- Specific change 1 with rationale
- Specific change 2 with rationale
- Specific change 3 with rationale

## Technical Approach
Explain any significant technical decisions or trade-offs made.

## Testing
- [x] Unit tests added/updated (X% coverage)
- [x] Integration tests cover new endpoints
- [x] E2E tests cover critical paths
- [x] Manual testing completed on all platforms
- [x] Performance impact assessed (no regression)

## Security Checklist
- [x] No PII exposed
- [x] Input validation implemented
- [x] Rate limiting in place
- [x] Child safety measures verified

## Breaking Changes
None OR describe migration path

## JIRA
[$ARGUMENTS](https://yqm0nk3y1.atlassian.net/browse/$ARGUMENTS)

## Screenshots/Videos
[Include for UI changes]

## Reviewer Notes
Specific areas that need careful review or testing.
EOF
)"

# Add reviewers based on CODEOWNERS
gh pr edit --add-reviewer @teamlead,@senior-dev
```

### 3. Update JIRA Ticket

```
1. Add PR link to ticket
2. Update status to "Review"
3. Add comment with testing instructions
4. Tag relevant stakeholders
```

### Collaboration Guidelines

1. **Daily Updates** (for multi-day tickets)
   - Progress made
   - Any blockers or concerns
   - Expected completion

2. **Architecture Decisions**
   - Discuss significant changes in team channel
   - Document decisions in ADRs
   - Get buy-in before major refactors

## Error Handling and Problem Resolution

### Common Issues and Solutions

#### 1. Build or Test Failures

```bash
# Clear all caches in feature worktree
npm run clean
rm -rf node_modules
npm install

# Check environment variables
env | grep -E "(API|DATABASE|SECRET)"

# Verify test database is running
docker-compose ps

# Debug specific test
npm test -- --watch path/to/test.ts
```

#### 2. Dependency Conflicts

```json
// Use resolutions in package.json
{
  "resolutions": {
    "conflicting-package": "1.2.3"
  }
}
```

#### 3. Cross-Platform Issues

```typescript
// Platform-specific handling
import { Platform } from 'react-native';

if (Platform.OS === 'ios') {
  // iOS specific code
} else if (Platform.OS === 'android') {
  // Android specific code
} else {
  // Web fallback
}
```

### Performance Troubleshooting

1. **React DevTools Profiler**
   - Identify unnecessary renders
   - Find performance bottlenecks
   - Optimize component hierarchies

2. **API Performance**
   ```typescript
   // Add timing logs
   const start = Date.now();
   const result = await apiCall();
   console.log(`API call took ${Date.now() - start}ms`);
   ```

3. **Bundle Size Analysis**
   ```bash
   npm run analyze-bundle
   ```

## Post-Merge Activities

### Immediate Actions

1. **Verify Deployment**
   - Check deployment pipeline
   - Monitor error tracking
   - Verify feature flags if used

2. **Update JIRA**
   - Transition to "Done"
   - Add implementation notes
   - Link to deployed version

3. **Clean Up Worktree**
   ```bash
   # Return to main worktree
   cd /path/to/main-project
   
   # Update main branch with latest changes
   git checkout main
   git pull origin main
   
   # Remove the feature worktree
   git worktree remove ../project-$ARGUMENTS
   
   # Clean up any remaining references
   git worktree prune
   
   # Optional: Delete the merged feature branch
   git branch -d feature/$ARGUMENTS-description
   git push origin --delete feature/$ARGUMENTS-description
   ```

### Follow-up Tasks

1. **Monitor Production**
   - Check error rates
   - Monitor performance metrics
   - Watch for user feedback

2. **Knowledge Sharing**
   - Update team wiki if needed
   - Share learnings in team channel
   - Schedule demo if significant feature

## Success Criteria Checklist

### Code Quality ✓
- [ ] All new and existing tests pass
- [ ] Test coverage ≥80% for new code
- [ ] Zero ESLint errors
- [ ] Full TypeScript type safety
- [ ] Performance benchmarks met
- [ ] Code follows team conventions

### Documentation ✓
- [ ] Code is self-documenting
- [ ] Complex logic has comments
- [ ] Public APIs have JSDoc
- [ ] README updated if needed
- [ ] JIRA has implementation notes


### Project Management ✓
- [ ] JIRA ticket in "Done"
- [ ] PR reviewed and merged
- [ ] No conflicts with main
- [ ] Stakeholders notified
- [ ] Feature worktree cleaned up

## Emergency Procedures

### Rollback Plan

If critical issues are discovered post-deployment:

1. **Immediate Rollback**
   ```bash
   # Create emergency worktree for hotfix
   git worktree add ../project-emergency-rollback main
   cd ../project-emergency-rollback
   
   # Revert via Git
   git revert -m 1 <merge-commit-hash>
   git push origin main
   ```

2. **Feature Flag Disable**
   ```typescript
   // If using feature flags
   featureFlags.disable('new-feature');
   ```

3. **Incident Documentation**
   - Create incident ticket
   - Document root cause
   - Plan fixes
   - Update runbook

### Escalation Path

1. **Severity Levels**
   - P0: Production down → Page on-call immediately
   - P1: Major feature broken → Notify team lead within 30min
   - P2: Minor issues → Create ticket for next sprint

2. **Communication**
   - Update #incidents channel
   - Create incident document
   - Schedule postmortem if P0/P1

## Git Worktree Benefits in This Workflow

### Key Advantages

1. **Parallel Development**
   - Continue working on other tickets while long-running tests execute
   - Switch between urgent hotfixes and feature development instantly
   - No need to stash/unstash work when context switching

2. **Isolated Testing**
   - Each worktree can run its own test suite independently
   - No interference between different development streams
   - Perfect for code reviews in separate environments

3. **Efficient Resource Usage**
   - Shared `.git` directory saves disk space vs multiple clones
   - All worktrees share the same Git history and configuration
   - Fast creation and deletion of working directories

4. **Emergency Response**
   - Instantly create hotfix worktrees without disrupting ongoing work
   - Maintain stable main worktree for production deployments
   - Quick rollback capabilities with dedicated emergency worktrees

### Managing Multiple Worktrees

```bash
# List all active worktrees
git worktree list

# Example output:
# /path/to/main-project          1a2b3c4 [main]
# /path/to/project-ADF-123       5d6e7f8 [feature/ADF-123-auth]
# /path/to/project-ADF-124       9a0b1c2 [feature/ADF-124-ui]
# /path/to/project-hotfix        3d4e5f6 [hotfix/critical-security]

# Remove completed worktrees
git worktree remove ../project-ADF-123

# Prune stale worktree references
git worktree prune
```

---

Remember: Quality over speed. It's better to take time to do it right than to rush and create technical debt or security issues. Git worktree enables this by removing the friction of context switching while maintaining code quality standards.