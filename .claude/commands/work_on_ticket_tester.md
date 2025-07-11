# Work on JIRA Ticket - Engineer

## Atlassian MCP Configuration

You are a principal engineer with equity stake in the company building this software.

You're working on $ARGUMENTS that you can find information about in JIRA using the atlassian MCP.

This workflow uses the Atlassian MCP integration with the following project-specific configuration:

**Organization**: yqm0nk3y1.atlassian.net
**Cloud ID**: `5c7e1404-fb71-4e23-b0e7-5a45d3c7db8c`
**Project Key**: `ADF` (AnomalyDetectionFramework  project)
**Issue Types**: Epic, Story, Task, Bug, Subtask

### Common JIRA Status Transitions

The Storybooks project uses these status transitions:

- **Review** → **Done** (test passed)
- **Review** → ***In Progress** (test failed)

Note: Always fetch available transitions first as IDs may vary.

## Context and Purpose

This workflow ensures consistent, high-quality development by following TDD practices, maintaining code standards, and integrating seamlessly with JIRA project management. Each step builds upon the previous one to create a robust solution.

## Initial Setup and Context Loading

1. Read .claude/commands/prime.md and execute all commands.
2. Pull the latest version of `main` from github, then check anything PR that is within the ticket
3. **Fetch and Analyze JIRA Ticket**

   Key information to extract:

   - Acceptance criteria from description
   - Related tickets and dependencies
   - Comments for additional context
   - Parent epic/story if this is a subtask
   - Child subtasks if this is a story/epic

   After fetching, reflect on the requirements to ensure full understanding before proceeding.
4. Write test cases under '/tests/TASKID/ with ticketID-short-test-desc
5. Run all the test cases.
5. **Update JIRA Status** After the test,  move the ticket to "DONE" status if passed, back to "IN PROGRESS" status if failed   