<script setup lang="ts">
const props = defineProps({
    class: {
        type: String,
    },
    layoutClass: {
        type: String,
    },
})
</script>

<template>
    <div class="slidev-layout two-columns w-full h-full grid grid-cols-2 gap-x-4 content-center grid-rows-[min-content,1fr]" :class="props.layoutClass">
        <h2 class="col-span-2">
            <slot name="title" />
        </h2>
        <div class="col-left" :class="props.class">
            <slot />
        </div>
        <div class="col-right" :class="props.class">
            <slot name="right" />
        </div>
    </div>
</template>