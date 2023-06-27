<script setup lang="ts">
const props = defineProps({
    class: {
        type: String,
    },
    layoutClass: {
        type: String,
    },
    title: {
        type: String,
        default: '',
    },
})
</script>

<template>
    <div class="slidev-layout two-columns w-full h-full grid grid-cols-2 gap-2" :class="props.layoutClass">
        <div class="col-left" :class="props.class">
            <slot />
        </div>
        <div class="col-right" :class="props.class">
            <slot name="right" />
        </div>
    </div>
</template>